import os
import re
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Set

import gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.requests import RequestsHTTPTransport
from humanfriendly import parse_size
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import CommonSettings

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
    auto_deploy_default_storage_provider=False,
    # specify initial amount of seconds to sleep before checking for job status
    init_seconds_before_status_checks=0,
)


class AuthenticationError(RuntimeError): ...


expr = re.compile(r"^(([a-zA-Z]+)://)?(?P<uri>[^/]+.*)$")


def sanitize_image_name(image: str) -> str:
    match = expr.match(image)
    if match is None:
        raise ValueError(f"malformed image name: {image}")

    return match["uri"]


@dataclass
class Resources:
    cpu: int = 2000  # millicores
    mem: int = 4 * 1024**3  # bytes
    disk: int = 500 * 1024**3  # bytes
    gpus: int = 0
    gpu_type: Optional[str] = None


resource_key_expr = re.compile(r"^(?P<type>mem|disk)_(?P<unit>\w+)$")


def get_resources(resources: dict[str, str]):
    res = Resources()

    for key, val in resources.items():
        if key.startswith("cpu"):
            if isinstance(val, int):
                res.cpu = val * 1000
            elif isinstance(val, str):
                if val.endswith("m"):
                    res.cpu = parse_size(val.strip("m"))
                else:
                    res.cpu = parse_size(val) * 1000

            continue

        match = resource_key_expr.match(key)
        if match is not None:
            unit = match["unit"]
            multiplier = parse_size(f"1 {unit}")

            setattr(res, match["type"], int(val) * multiplier)

            continue

        if key == "gpu" or key == "gpus":
            res.gpus = int(val)

        if key == "gpu_type":
            res.gpu_type = val

    # todo(ayush): do some validation here

    return res


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

        job_exec = self.format_job_exec(job)

        # strip leading python3 -m bc path to python is absolute in the runtime task and not
        # necessarily available to the job machine
        command = job_exec.split()[2:]

        if "--quiet" not in command:
            command.extend(["--quiet", "all"])

        image: Optional[str] = getattr(job, "container_img_url")
        if image is None:
            image = job.resources.get("container")
        if image is None:
            image = getattr(self.workflow.remote_execution_settings, "container_image")
        if image is None:
            raise ValueError(
                f"{rule} - {job.jobid}: rule must have a container image configured to run on latch"
            )

        image = sanitize_image_name(image)

        resources = get_resources(job.resources)

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
