import os
from typing import AsyncGenerator, List, Optional, Set

import gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.requests import RequestsHTTPTransport
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import CommonSettings

from .resource_utils import (
    get_resources,
    sanitize_image_name,
    validate_and_pin_gpu_resources,
    validate_resource_limits,
)

# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    # define whether your executor plugin executes locally
    # or remotely. In virtually all cases, it will be remote execution
    # (cluster, cloud, etc.). Only Snakemake's standard execution
    # plugins (snakemake-executor-plugin-dryrun, snakemake-executor-plugin-local)
    # are expected to specify False here.
    non_local_exec=True,
    # Whether the executor implies to not have a shared file system
    implies_no_shared_fs=False,  # -- we will be using OFS
    # whether to deploy workflow sources to default storage provider before execution
    job_deploy_sources=False,
    # whether arguments for setting the storage provider shall be passed to jobs
    pass_default_storage_provider_args=True,
    # whether arguments for setting default resources shall be passed to jobs
    pass_default_resources_args=False,
    # whether environment variables shall be passed to jobs (if False, use
    # self.envvars() to obtain a dict of environment variables and their values
    # and pass them e.g. as secrets to the execution backend)
    pass_envvar_declarations_to_cmd=True,
    # whether the default storage provider shall be deployed before the job is run on
    # the remote node. Usually set to True if the executor does not assume a shared fs
    auto_deploy_default_storage_provider=False,
    # specify initial amount of seconds to sleep before checking for job status
    init_seconds_before_status_checks=0,
)


class AuthenticationError(RuntimeError): ...


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def __post_init__(self):
        # access workflow
        # self.workflow
        # access executor specific settings
        # self.workflow.execution_settings

        # IMPORTANT: in your plugin, only access methods and properties of
        # Snakemake objects (like Workflow, Persistence, etc.) that are
        # defined in the interfaces found in the
        # snakemake-interface-executor-plugins and the
        # snakemake-interface-common package.
        # Other parts of those objects are NOT guaranteed to remain
        # stable across new releases.

        # To ensure that the used interfaces are not changing, you should
        # depend on these packages as >=a.b.c,<d with d=a+1 (i.e. pin the
        # dependency on this package to be at least the version at time
        # of development and less than the next major version which would
        # introduce breaking changes).

        # In case of errors outside of jobs, please raise a WorkflowError
        auth_header: Optional[str] = None

        self.execution_token = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "")
        if self.execution_token == "":
            raise AuthenticationError(
                "Unable to find credentials to connect to gql server, aborting"
            )

        auth_header = f"Latch-Execution-Token {self.execution_token}"

        domain = os.environ.get("LATCH_SDK_DOMAIN", "latch.bio")
        url = f"https://vacuole.{domain}/graphql"

        self.sync_gql_session = gql.Client(
            transport=RequestsHTTPTransport(
                url=url, headers={"Authorization": auth_header}, retries=5, timeout=90
            )
        )

        self.async_gql_session = gql.Client(
            transport=AIOHTTPTransport(
                url=url, headers={"Authorization": auth_header}, timeout=90
            )
        )

    def run_job(self, job: JobExecutorInterface):
        rule = next(x for x in job.rules)

        job_exec = self.format_job_exec(job)

        # strip leading python3 -m bc path to python is absolute in the runtime task and not
        # necessarily available to the job machine
        command = job_exec.split()[2:]

        if (
            os.environ.get("LATCH_VERBOSE_RULE_OUTPUT") is None
            and "--quiet" not in command
        ):
            command.extend(["--quiet", "all"])

        image = os.environ.get("FLYTE_INTERNAL_IMAGE")  # always set during register
        image = job.resources.get("container", image)
        image = sanitize_image_name(image)

        resources = get_resources(job.resources)
        if resources.gpu_type is not None:
            if resources.gpus == 0:
                self.logger.info(
                    f"Ignoring `gpu_type` resource on rule {rule} as it requests 0 GPUs"
                )
                resources.gpu_type = None
            else:
                resources = validate_and_pin_gpu_resources(resources)
                self.logger.info(
                    f"As `gpu_type` is provided and `gpus` > 0, resources have been overridden to {resources}"
                )

        validate_resource_limits(resources)

        if resources.gpus > 0 and resources.gpu_type is None:
            raise ValueError(
                f"{rule} - {job.jobid}: rule that requests gpus must also specify a gpu type"
            )

        upstream: Set[int] = set()
        for dep in self.dag.dependencies[job]:
            upstream.add(dep.jobid)

        self.sync_gql_session.execute(
            gql.gql(
                """
                mutation CreateJob(
                    $argExecutionToken: String!
                    $argRule: String!
                    $argJobId: BigInt!
                    $argImage: String!
                    $argCommand: [String!]!
                    $argCpuLimitMillicores: BigInt!
                    $argMemoryLimitBytes: BigInt!
                    $argEphemeralStorageLimitBytes: BigInt!
                    $argGpuLimit: BigInt!
                    $argGpuType: String
                    $argParentJobIds: [BigInt!]!
                    $argAttempt: BigInt!
                ) {
                    smCreateJob(
                        input: {
                            argExecutionToken: $argExecutionToken
                            argRule: $argRule
                            argJobId: $argJobId
                            argImage: $argImage
                            argCommand: $argCommand
                            argCpuLimitMillicores: $argCpuLimitMillicores
                            argMemoryLimitBytes: $argMemoryLimitBytes
                            argEphemeralStorageLimitBytes: $argEphemeralStorageLimitBytes
                            argGpuLimit: $argGpuLimit
                            argGpuType: $argGpuType
                            argParentJobIds: $argParentJobIds
                            argAttempt: $argAttempt
                        }
                    ) {
                        clientMutationId
                    }
                }
                """
            ),
            {
                "argExecutionToken": self.execution_token,
                "argRule": rule,
                "argJobId": job.jobid,
                "argCommand": command,
                "argImage": image,
                "argCpuLimitMillicores": resources.cpu,
                "argMemoryLimitBytes": resources.mem,
                "argEphemeralStorageLimitBytes": resources.disk,
                "argGpuLimit": resources.gpus,
                "argGpuType": resources.gpu_type,
                "argParentJobIds": list(upstream),
                "argAttempt": job.attempt,
            },
        )

        self.report_job_submission(SubmittedJobInfo(job))

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:
        job_infos_by_id = {x.job.jobid: x for x in active_jobs}

        statuses = (
            await self.async_gql_session.execute_async(
                gql.gql(
                    """
                    query CheckActiveJobs($argExecutionToken: String!, $argJobIds: [BigInt!]!) {
                        smMultiGetJobInfos(
                            argExecutionToken: $argExecutionToken
                            argJobIds: $argJobIds
                        ) {
                            nodes {
                                jobId
                                latestExecution {
                                    status
                                    errorMessage
                                }
                            }
                        }
                    }
                    """
                ),
                {
                    "argExecutionToken": self.execution_token,
                    "argJobIds": [x.job.jobid for x in active_jobs],
                },
            )
        )["smMultiGetJobInfos"]

        for node in statuses["nodes"]:
            exec = node["latestExecution"]
            status = exec["status"]

            job_info = job_infos_by_id[int(node["jobId"])]

            if status in {"SUCCEEDED", "SKIPPED"}:
                self.report_job_success(job_info)
            elif status in {"FAILED", "ABORTED"}:
                self.report_job_error(job_info, exec["errorMessage"])
            else:
                yield job_info

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        # Cancel all active jobs.
        # This method is called when Snakemake is interrupted.
        job_ids = []
        for job_info in active_jobs:
            job_ids.append(job_info.job.jobid)

        self.sync_gql_session.execute(
            gql.gql(
                """
                mutation CancelJobs($argJobIds: [BigInt!]!, $argExecutionToken: String = "") {
                    smMultiCancelJobs(
                        input: {
                            argJobIds: $argJobIds
                            argExecutionToken: $argExecutionToken
                        }
                    ) {
                        clientMutationId
                    }
                }
                """,
                {
                    "argJobIds": job_ids,
                    "argExecutionToken": self.execution_token,
                },
            )
        )
