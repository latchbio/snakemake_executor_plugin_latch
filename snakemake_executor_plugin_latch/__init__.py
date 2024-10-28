import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Optional

import gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.requests import RequestsHTTPTransport
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.logging import LoggerExecutorInterface
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecutorSettingsBase,
)
from snakemake_interface_executor_plugins.workflow import WorkflowExecutorInterface


# Optional:
# Define additional settings for your executor.
# They will occur in the Snakemake CLI as --<executor-name>-<param-name>
# Omit this class if you don't need any.
# Make sure that all defined fields are Optional and specify a default value
# of None or anything else that makes sense in your case.
@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    pvc_name: str = field(
        default="",
        metadata={
            "help": "The PVC for the shared workdir",
            "required": True,
        },
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
    job_deploy_sources=True,  # todo(ayush): is this necessary
    # whether arguments for setting the storage provider shall be passed to jobs
    pass_default_storage_provider_args=True,
    # whether arguments for setting default resources shall be passed to jobs
    pass_default_resources_args=True,
    # whether environment variables shall be passed to jobs (if False, use
    # self.envvars() to obtain a dict of environment variables and their values
    # and pass them e.g. as secrets to the execution backend)
    pass_envvar_declarations_to_cmd=True,
    # whether the default storage provider shall be deployed before the job is run on
    # the remote node. Usually set to True if the executor does not assume a shared fs
    auto_deploy_default_storage_provider=True,
    # specify initial amount of seconds to sleep before checking for job status
    init_seconds_before_status_checks=0,
)


class AuthenticationError(RuntimeError): ...


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def _get_execution_token(self):
        token = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "")
        token = "f48a8b3ccaeba488cb35"
        if token == "":
            raise RuntimeError("unable to get current execution token")

        return token

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

        token = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "")
        if token != "":
            auth_header = f"Latch-Execution-Token {token}"

        if auth_header is None:
            token_path = Path.home() / ".latch" / "dev-token"
            if token_path.exists():
                auth_header = f"Latch-SDK-Token {token_path.read_text().strip()}"

        if auth_header is None:
            raise AuthenticationError(
                "Unable to find credentials to connect to gql server, aborting"
            )

        domain = os.environ.get("LATCH_SDK_DOMAIN", "latch.bio")
        domain = "ligma.ai"

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
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.
        # If required, make sure to pass the job's id to the job_info object, as keyword
        # argument 'external_job_id'.

        rule = next(x for x in job.rules)

        token = self._get_execution_token()

        command = ["/bin/bash", "-c", self.format_job_exec(job)]

        image = job.resources.get("container")
        if image is None:
            image = getattr(self.workflow.remote_execution_settings, "container_image")
        if image is None:
            raise ValueError(
                f"{rule} - {job.jobid}: rule must have a container image configured to run on latch"
            )

        cpu = int(float(job.resources["_cores"]) * 1000)
        # default 512 MiB
        ram = int(float(job.resources.get("mem_mb", 512)) * 1024 * 1024)
        # default 512 GiB
        disk = int(float(job.resources.get("disk_mb", 512 * 1024)) * 1024 * 1024)

        gpus = int(job.resources.get("gpu", 0))
        gpu_type = job.resources.get("gpu_type")

        if gpus > 0 and gpu_type is None:
            raise ValueError(
                f"{rule} - {job.jobid}: rule that requests gpus must also specify a gpu type"
            )

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
                ) {
                    smCreateJobInfo(
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
                        }
                    ) {
                        clientMutationId
                    }
                }
                """
            ),
            {
                "argExecutionToken": token,
                "argRule": rule,
                "argJobId": job.jobid,
                "argCommand": command,
                "argImage": image,
                "argCpuLimitMillicores": cpu,
                "argMemoryLimitBytes": ram,
                "argEphemeralStorageLimitBytes": disk,
                "argGpuLimit": gpus,
                "argGpuType": gpu_type,
            },
        )

        self.report_job_submission(SubmittedJobInfo(job))

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:

        # Check the status of active jobs.

        # You have to iterate over the given list active_jobs.
        # If you provided it above, each will have its external_jobid set according
        # to the information you provided at submission time.
        # For jobs that have finished successfully, you have to call
        # self.report_job_success(active_job).
        # For jobs that have errored, you have to call
        # self.report_job_error(active_job).
        # This will also take care of providing a proper error message.
        # Usually there is no need to perform additional logging here.
        # Jobs that are still running have to be yielded.
        #
        # For queries to the remote middleware, please use
        # self.status_rate_limiter like this:
        #
        # async with self.status_rate_limiter:
        #    # query remote middleware here
        #
        # To modify the time until the next call of this method,
        # you can set self.next_sleep_seconds here.

        token = self._get_execution_token()

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
                                status
                                errorMessage
                            }
                        }
                    }
                    """
                ),
                {
                    "argExecutionToken": token,
                    "argJobIds": [x.job.jobid for x in active_jobs],
                },
            )
        )["smMultiGetJobInfos"]

        for node in statuses["nodes"]:
            status = node["status"]
            job_info = job_infos_by_id[int(node["jobId"])]

            if status in {"SUCCEEDED", "SKIPPED"}:
                self.report_job_success(job_info)
            elif status in {"FAILED", "ABORTED"}:
                self.report_job_error(job_info, node["errorMessage"])
            else:
                yield job_info

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        # Cancel all active jobs.
        # This method is called when Snakemake is interrupted.
        return
