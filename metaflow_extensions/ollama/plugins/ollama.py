import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import socket
import sys
import os
import functools
import json
import requests

from .constants import OLLAMA_SUFFIX
from .exceptions import (
    EmptyOllamaManifestCacheException,
    EmptyOllamaBlobCacheException,
    UnspecifiedRemoteStorageRootException,
)


class ProcessStatus:
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCESSFUL = "SUCCESSFUL"


class OllamaManager:
    """
    A process manager for Ollama runtimes.
    Implements interface @ollama([models=...], ...) has a local, remote, or managed backend.
    """

    def __init__(
        self,
        models,
        backend="local",
        flow_datastore_backend=None,
        remote_storage_root=None,
        force_pull=False,
        skip_push_check=False,
        debug=False,
    ):
        self.models = {}
        self.processes = {}
        self.flow_datastore_backend = flow_datastore_backend
        if self.flow_datastore_backend is not None:
            self.remote_storage_root = self.get_ollama_storage_root(
                self.flow_datastore_backend
            )
        elif remote_storage_root is not None:
            self.remote_storage_root = remote_storage_root
        else:
            raise UnspecifiedRemoteStorageRootException(
                "Can not determine the storage root, as both flow_datastore_backend and remote_storage_root arguments of OllamaManager are None."
            )
        self.force_pull = force_pull
        self.skip_push_check = skip_push_check
        self.debug = debug
        self.stats = {}
        self.storage_info = {}
        self.ollama_url = "http://localhost:11434"  # Ollama API base URL

        if backend != "local":
            raise ValueError(
                "OllamaManager only supports the 'local' backend at this time."
            )

        self._timeit(self._install_ollama, "install_ollama")
        self._timeit(self._launch_server, "launch_server")

        # Pull models concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._pull_model, m) for m in models]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise RuntimeError(f"Error pulling one or more models. {e}") from e

        # Run models as background processes.
        for m in models:
            f = functools.partial(self._run_model, m)
            self._timeit(f, f"model_{m.lower()}")

    def _timeit(self, f, name):
        t0 = time.time()
        f()
        tf = time.time()
        self.stats[name] = {"process_runtime": tf - t0}

    def _install_ollama(self, max_retries=3):
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                print("[@ollama] Ollama is already installed.")
                return
        except Exception as e:
            if self.debug:
                print(f"[@ollama] Did not find Ollama installation: {e}")
            if sys.platform == "darwin":
                raise RuntimeError(
                    "On macOS, please install Ollama manually from https://ollama.com/download."
                )

        if self.debug:
            print("[@ollama] Installing Ollama...")
        env = os.environ.copy()
        env["CURL_IPRESOLVE"] = "4"

        for attempt in range(max_retries):
            try:
                install_cmd = ["curl", "-fsSL", "https://ollama.com/install.sh"]
                curl_proc = subprocess.run(
                    install_cmd, capture_output=True, text=True, env=env
                )
                if curl_proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to download Ollama install script: stdout: {curl_proc.stdout}, stderr: {curl_proc.stderr}"
                    )
                sh_proc = subprocess.run(
                    ["sh"],
                    input=curl_proc.stdout,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if sh_proc.returncode != 0:
                    raise RuntimeError(
                        f"Ollama installation script failed: stdout: {sh_proc.stdout}, stderr: {sh_proc.stderr}"
                    )
                if self.debug:
                    print("[@ollama] Ollama installed successfully.")
                break
            except Exception as e:
                if self.debug:
                    print(f"[@ollama] Installation attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Error installing Ollama after {max_retries} attempts: {e}"
                    ) from e

    def _is_port_open(self, host, port, timeout=1):
        """Check if a TCP port is open on a given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                return True
            except socket.error:
                return False

    def _launch_server(self):
        """
        Start the Ollama server process and ensure it's running.
        """
        try:
            print("[@ollama] Starting Ollama server...")
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes[process.pid] = {
                "p": process,
                "properties": {"type": "api-server", "error_details": None},
                "status": ProcessStatus.RUNNING,
            }

            if self.debug:
                print(f"[@ollama] Started server process with PID {process.pid}.")

            # Wait until the server is ready
            host, port = "127.0.0.1", 11434
            retries = 0
            max_retries = 10
            while (
                not self._is_port_open(host, port, timeout=1) and retries < max_retries
            ):
                if retries == 0:
                    print("[@ollama] Waiting for server to be ready...")
                elif retries % 3 == 0:
                    print(f"[@ollama] Still waiting... ({retries + 1}/{max_retries})")
                time.sleep(5)
                retries += 1

            if not self._is_port_open(host, port, timeout=1):
                error_details = (
                    f"Ollama server did not start listening on {host}:{port}"
                )
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            # Check if process terminated unexpectedly
            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                error_details = f"Return code: {returncode}, Error: {stderr}"
                self.processes[process.pid]["properties"][
                    "error_details"
                ] = error_details
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                raise RuntimeError(f"Ollama server failed to start. {error_details}")

            print("[@ollama] Server is ready.")

        except Exception as e:
            if "process" in locals() and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            raise RuntimeError(f"Error starting Ollama server: {e}") from e

    def _setup_storage(self, m):
        """
        Configure local and remote storage paths for an Ollama model.
        """
        # Parse model and tag name
        ollama_model_name_components = m.split(":")
        if len(ollama_model_name_components) == 1:
            model_name = ollama_model_name_components[0]
            tag = "latest"
        elif len(ollama_model_name_components) == 2:
            model_name = ollama_model_name_components[0]
            tag = ollama_model_name_components[1]

        # Find where Ollama actually stores models
        possible_storage_roots = [
            os.environ.get("OLLAMA_MODELS"),
            "/usr/share/ollama/.ollama/models",
            os.path.expanduser("~/.ollama/models"),
            "/root/.ollama/models",
        ]

        ollama_local_storage_root = None
        for root in possible_storage_roots:
            if root and os.path.exists(root):
                ollama_local_storage_root = root
                break

        if not ollama_local_storage_root:
            # https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored
            if sys.platform.startswith("linux"):
                ollama_local_storage_root = "/usr/share/ollama/.ollama/models"
            elif sys.platform == "darwin":
                ollama_local_storage_root = os.path.expanduser("~/.ollama/models")

        if self.debug:
            print(
                f"[@ollama {m}] Using Ollama storage root: {ollama_local_storage_root}."
            )

        blob_local_path = os.path.join(ollama_local_storage_root, "blobs")
        manifest_base_path = os.path.join(
            ollama_local_storage_root,
            "manifests/registry.ollama.ai/library",
            model_name,
        )

        # Create directories
        try:
            os.makedirs(blob_local_path, exist_ok=True)
            os.makedirs(manifest_base_path, exist_ok=True)
        except FileExistsError:
            pass

        # Set up remote paths
        if not self.local_datastore and self.remote_storage_root is not None:
            blob_remote_key = os.path.join(self.remote_storage_root, "blobs")
            manifest_remote_key = os.path.join(
                self.remote_storage_root,
                "manifests/registry.ollama.ai/library",
                model_name,
                tag,
            )
        else:
            blob_remote_key = None
            manifest_remote_key = None

        self.storage_info[m] = {
            "blob_local_root": blob_local_path,
            "blob_remote_root": blob_remote_key,
            "manifest_local": os.path.join(manifest_base_path, tag),
            "manifest_remote": manifest_remote_key,
            "manifest_content": None,
            "model_name": model_name,
            "tag": tag,
            "storage_root": ollama_local_storage_root,
        }

        if self.debug:
            print(f"[@ollama {m}] Storage paths configured.")

    def _fetch_manifest(self, m):
        """
        Load the manifest file and content, either from local storage or remote cache.
        """
        if self.debug:
            print(f"[@ollama {m}] Checking for cached manifest...")

        def _disk_to_memory():
            with open(self.storage_info[m]["manifest_local"], "r") as f:
                self.storage_info[m]["manifest_content"] = json.load(f)

        if os.path.exists(self.storage_info[m]["manifest_local"]):
            if self.storage_info[m]["manifest_content"] is None:
                _disk_to_memory()
            if self.debug:
                print(f"[@ollama {m}] Manifest found locally.")
        elif self.local_datastore:
            if self.debug:
                print(f"[@ollama {m}] No manifest found in local datastore.")
            return None
        else:
            from metaflow import S3
            from metaflow.plugins.datatools.s3.s3 import MetaflowS3NotFound

            try:
                with S3() as s3:
                    s3obj = s3.get(self.storage_info[m]["manifest_remote"])
                    if not s3obj.exists:
                        raise EmptyOllamaManifestCacheException(
                            f"No manifest in remote storage for model {m}"
                        )

                    if self.debug:
                        print(f"[@ollama {m}] Downloaded manifest from cache.")
                    os.rename(s3obj.path, self.storage_info[m]["manifest_local"])
                    _disk_to_memory()

                    if self.debug:
                        print(
                            f"[@ollama {m}] Manifest found in remote cache, downloaded locally."
                        )
            except (MetaflowS3NotFound, EmptyOllamaManifestCacheException):
                if self.debug:
                    print(
                        f"[@ollama {m}] No manifest found locally or in remote cache."
                    )
                return None

        return self.storage_info[m]["manifest_content"]

    def _fetch_blobs(self, m):
        """
        Fetch missing blobs from remote cache.
        """
        if self.debug:
            print(f"[@ollama {m}] Checking for cached blobs...")

        manifest = self._fetch_manifest(m)
        if not manifest:
            raise EmptyOllamaBlobCacheException(f"No manifest available for model {m}")

        blobs_required = [layer["digest"] for layer in manifest["layers"]]
        missing_blob_info = []

        # Check which blobs are missing locally
        for blob_digest in blobs_required:
            blob_filename = blob_digest.replace(":", "-")
            local_blob_path = os.path.join(
                self.storage_info[m]["blob_local_root"], blob_filename
            )

            if not os.path.exists(local_blob_path):
                if self.debug:
                    print(f"[@ollama {m}] Blob {blob_digest} not found locally.")

                remote_blob_path = os.path.join(
                    self.storage_info[m]["blob_remote_root"], blob_filename
                )
                missing_blob_info.append(
                    {
                        "digest": blob_digest,
                        "filename": blob_filename,
                        "remote_path": remote_blob_path,
                        "local_path": local_blob_path,
                    }
                )

        if not missing_blob_info:
            if self.debug:
                print(f"[@ollama {m}] All blobs found locally.")
            return

        if self.debug:
            print(
                f"[@ollama {m}] Downloading {len(missing_blob_info)} missing blobs from cache..."
            )

        remote_urls = [blob_info["remote_path"] for blob_info in missing_blob_info]

        from metaflow import S3

        try:
            with S3() as s3:
                if len(remote_urls) == 1:
                    s3objs = [s3.get(remote_urls[0])]
                else:
                    s3objs = s3.get_many(remote_urls)

                if not isinstance(s3objs, list):
                    s3objs = [s3objs]

                # Move each downloaded blob to correct location
                for i, s3obj in enumerate(s3objs):
                    if not s3obj.exists:
                        blob_info = missing_blob_info[i]
                        raise EmptyOllamaBlobCacheException(
                            f"Blob {blob_info['digest']} not found in remote cache for model {m}"
                        )

                    blob_info = missing_blob_info[i]
                    os.makedirs(os.path.dirname(blob_info["local_path"]), exist_ok=True)
                    os.rename(s3obj.path, blob_info["local_path"])

                    if self.debug:
                        print(f"[@ollama {m}] Downloaded blob {blob_info['filename']}.")

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error during blob fetch: {e}")
            raise EmptyOllamaBlobCacheException(
                f"Failed to fetch blobs for model {m}: {e}"
            )

        if self.debug:
            print(
                f"[@ollama {m}] Successfully downloaded all missing blobs from cache."
            )

    def _verify_model_available(self, m):
        """
        Verify model is available using Ollama API
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/show", json={"model": m}, timeout=10
            )

            available = response.status_code == 200

            if self.debug:
                if available:
                    print(f"[@ollama {m}] ✓ Model is available via API.")
                else:
                    print(
                        f"[@ollama {m}] ✗ Model not available via API (status: {response.status_code})."
                    )

            return available

        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error verifying model: {e}")
            return False

    def _register_cached_model_with_ollama(self, m):
        """
        Register a cached model with Ollama using the API.
        """
        try:
            show_response = requests.post(
                f"{self.ollama_url}/api/show", json={"model": m}, timeout=10
            )

            if show_response.status_code == 200:
                if self.debug:
                    print(f"[@ollama {m}] Model already registered with Ollama.")
                return True

            # Try to create/register the model from existing files
            if self.debug:
                print(f"[@ollama {m}] Registering cached model with Ollama...")

            create_response = requests.post(
                f"{self.ollama_url}/api/create",
                json={
                    "model": m,
                    "from": m,  # Use same name - should find existing files
                    "stream": False,
                },
                timeout=60,
            )

            if create_response.status_code == 200:
                result = create_response.json()
                if result.get("status") == "success":
                    if self.debug:
                        print(f"[@ollama {m}] Successfully registered cached model.")
                    return True
                else:
                    if self.debug:
                        print(f"[@ollama {m}] Create response: {result}.")

            # Fallback: try a pull which should be fast if files exist
            if self.debug:
                print(f"[@ollama {m}] Create failed, trying pull to register...")

            pull_response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"model": m, "stream": False},
                timeout=120,
            )

            if pull_response.status_code == 200:
                result = pull_response.json()
                if result.get("status") == "success":
                    if self.debug:
                        print(f"[@ollama {m}] Model registered via pull.")
                    return True

        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"[@ollama {m}] API registration failed: {e}")
        except Exception as e:
            if self.debug:
                print(f"[@ollama {m}] Error during registration: {e}")

        return False

    def _pull_model(self, m):
        """
        Pull/setup a model, using cache when possible.
        """
        self._setup_storage(m)

        # Try to fetch manifest from cache first
        manifest = None
        try:
            manifest = self._fetch_manifest(m)
        except (EmptyOllamaManifestCacheException, Exception) as e:
            if self.debug:
                print(f"[@ollama {m}] No cached manifest found or error fetching: {e}")
            manifest = None

        # If we don't have a cached manifest or force_pull is True, pull the model
        if self.force_pull or not manifest:
            try:
                print(f"[@ollama {m}] Not using cache. Downloading model {m}...")
                result = subprocess.run(
                    ["ollama", "pull", m], capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to pull model {m}: stdout: {result.stdout}, stderr: {result.stderr}"
                    )
                print(f"[@ollama {m}] Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Error pulling Ollama model {m}: {e}") from e
        else:
            # We have a cached manifest, try to fetch the blobs
            try:
                self._fetch_blobs(m)
                print(f"[@ollama {m}] Using cached model.")

                # Register the cached model with Ollama
                if not self._verify_model_available(m):
                    if not self._register_cached_model_with_ollama(m):
                        raise RuntimeError(
                            f"Failed to register cached model {m} with Ollama"
                        )

                # self.skip_push_check = True

            except (EmptyOllamaBlobCacheException, Exception) as e:
                if self.debug:
                    print(f"[@ollama {m}] Cache failed, downloading model...")
                    print(f"[@ollama {m}] Error: {e}")

                # Fallback to pulling the model
                try:
                    result = subprocess.run(
                        ["ollama", "pull", m], capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"Failed to pull model {m}: stdout: {result.stdout}, stderr: {result.stderr}"
                        )
                    print(f"[@ollama {m}] Model downloaded successfully (fallback).")
                except Exception as pull_e:
                    raise RuntimeError(
                        f"Error pulling Ollama model {m} as fallback: {pull_e}"
                    ) from pull_e

        # Final verification that the model is available
        if not self._verify_model_available(m):
            raise RuntimeError(f"Model {m} is not available to Ollama after setup")

        if self.debug:
            print(f"[@ollama {m}] Model setup complete and verified.")

    def _run_model(self, m):
        """
        Start the Ollama model as a subprocess and record its status.
        """
        process = None
        try:
            if self.debug:
                print(f"[@ollama {m}] Starting model process...")

            process = subprocess.Popen(
                ["ollama", "run", m],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes[process.pid] = {
                "p": process,
                "properties": {"type": "model", "model": m, "error_details": None},
                "status": ProcessStatus.RUNNING,
            }

            if self.debug:
                print(f"[@ollama {m}] Model process PID: {process.pid}.")

            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                if returncode == 0:
                    self.processes[process.pid]["status"] = ProcessStatus.SUCCESSFUL
                    if self.debug:
                        print(
                            f"[@ollama {m}] Process {process.pid} exited successfully."
                        )
                else:
                    error_details = f"Return code: {returncode}, Error: {stderr}"
                    self.processes[process.pid]["properties"][
                        "error_details"
                    ] = error_details
                    self.processes[process.pid]["status"] = ProcessStatus.FAILED
                    if self.debug:
                        print(
                            f"[@ollama {m}] Process {process.pid} failed: {error_details}."
                        )
        except Exception as e:
            if process and process.pid in self.processes:
                self.processes[process.pid]["status"] = ProcessStatus.FAILED
                self.processes[process.pid]["properties"]["error_details"] = str(e)
            raise RuntimeError(f"Error running Ollama model {m}: {e}") from e

    def terminate_models(self, skip_push_check=None):
        """
        Terminate all processes gracefully and update cache.
        """
        print("[@ollama] Shutting down models...")

        if skip_push_check is not None:
            assert isinstance(
                skip_push_check, bool
            ), "skip_push_check passed to terminate_models must be a bool if specified."
            self.skip_push_check = skip_push_check

        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "model":
                model_name = process_info["properties"].get("model")

                if self.debug:
                    print(f"[@ollama {model_name}] Stopping model process...")

                try:
                    result = subprocess.run(
                        ["ollama", "stop", model_name], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        process_info["status"] = ProcessStatus.SUCCESSFUL
                        if self.debug:
                            print(f"[@ollama {model_name}] Stopped successfully.")
                    else:
                        process_info["status"] = ProcessStatus.FAILED
                        if self.debug:
                            print(
                                f"[@ollama {model_name}] Stop failed: {result.stderr}"
                            )
                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    print(f"[@ollama {model_name}] Error stopping: {e}")

                # Update cache if needed
                if not self.skip_push_check:
                    self._update_model_cache(model_name)

        # Stop the API server
        for pid, process_info in list(self.processes.items()):
            if process_info["properties"].get("type") == "api-server":
                if self.debug:
                    print(f"[@ollama] Stopping API server process PID {pid}.")

                process = process_info["p"]
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(
                            f"[@ollama] API server PID {pid} did not terminate, killing..."
                        )
                        process.kill()
                        process.wait()

                    process_info["status"] = ProcessStatus.SUCCESSFUL
                    if self.debug:
                        print(f"[@ollama] API server terminated successfully.")
                except Exception as e:
                    process_info["status"] = ProcessStatus.FAILED
                    print(f"[@ollama] Warning: Error terminating API server: {e}")

        print("[@ollama] All models stopped.")

        # Show performance summary
        if self.debug:
            if hasattr(self, "stats") and self.stats:
                print("[@ollama] Performance summary:")
                for operation, stats in self.stats.items():
                    runtime = stats.get("process_runtime", 0)
                    if runtime > 1:  # Only show operations that took meaningful time
                        print(f"[@ollama]   {operation}: {runtime:.1f}s")

    def _update_model_cache(self, model_name):
        """
        Update the remote cache with model files if needed.
        """
        try:
            manifest = self._fetch_manifest(model_name)
            if not manifest:
                if self.debug:
                    print(
                        f"[@ollama {model_name}] No manifest available for cache update."
                    )
                return

            from metaflow import S3

            cache_up_to_date = True
            key_paths = [
                (
                    self.storage_info[model_name]["manifest_remote"],
                    self.storage_info[model_name]["manifest_local"],
                )
            ]

            with S3() as s3:
                # Check if blobs need updating
                s3objs = s3.list_paths(
                    [self.storage_info[model_name]["blob_remote_root"]]
                )
                for layer in manifest["layers"]:
                    expected_blob_sha = layer["digest"]
                    if expected_blob_sha not in s3objs:
                        cache_up_to_date = False
                        break

                if not cache_up_to_date:
                    blob_count = len(manifest.get("layers", []))
                    print(
                        f"[@ollama {model_name}] Uploading {blob_count} files to cache..."
                    )

                    # Add blob paths to upload
                    for layer in manifest["layers"]:
                        blob_filename = layer["digest"].replace(":", "-")
                        key_paths.append(
                            (
                                os.path.join(
                                    self.storage_info[model_name]["blob_remote_root"],
                                    blob_filename,
                                ),
                                os.path.join(
                                    self.storage_info[model_name]["blob_local_root"],
                                    blob_filename,
                                ),
                            )
                        )

                    s3.put_files(key_paths)
                    print(f"[@ollama {model_name}] Cache updated.")
                else:
                    if self.debug:
                        print(f"[@ollama {model_name}] Cache is up to date.")

        except Exception as e:
            if self.debug:
                print(f"[@ollama {model_name}] Error updating cache: {e}")

    def get_ollama_storage_root(self, backend):
        """
        Return the path to the root of the datastore.
        """
        if backend.TYPE == "s3":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_S3, OLLAMA_SUFFIX)
        elif backend.TYPE == "azure":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_AZURE, OLLAMA_SUFFIX)
        elif backend.TYPE == "gs":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_GS

            self.local_datastore = False
            return os.path.join(DATASTORE_SYSROOT_GS, OLLAMA_SUFFIX)
        else:
            self.local_datastore = True
            return None
