# from metaflow import FlowSpec, step, ollama, kubernetes, pypi
# from metaflow.profilers import gpu_profile
# import time
# import os

# class DeepSeekDebugTest(FlowSpec):

#     @pypi(packages={"ollama": "0.5.1"})
#     @kubernetes(cpu=32, gpu=4, memory=164_000, shared_memory=128_000, compute_pool="a10g4x")
#     @gpu_profile(interval=1)
#     @ollama(
#         models=['deepseek-r1:70b'],
#         debug=True,  # Enable debug logging
#         circuit_breaker_config={
#             'failure_threshold': 3,
#             'recovery_timeout': 30,
#             'reset_timeout': 60
#         }
#     )
#     @step
#     def start(self):
#         # Configure Ollama for multi-GPU usage
#         os.environ['OLLAMA_NUM_PARALLEL'] = '4'  # Use all 4 GPUs
#         os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Keep one model loaded
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Ensure all GPUs visible

#         print("[@test] Starting DeepSeek debug test...")
#         print(f"[@test] Environment: OLLAMA_NUM_PARALLEL={os.environ.get('OLLAMA_NUM_PARALLEL')}")
#         print(f"[@test] Environment: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

#         # Import inside step to ensure protection is installed
#         from ollama import chat
#         import subprocess

#         # Debug: Check GPU allocation
#         try:
#             result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
#             print("[@test] GPU Status:")
#             print(result.stdout)
#         except Exception as e:
#             print(f"[@test] Could not check GPU status: {e}")

#         # Debug: Check Ollama configuration
#         try:
#             result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
#             print("[@test] Ollama models list:")
#             print(result.stdout)
#         except Exception as e:
#             print(f"[@test] Could not list ollama models: {e}")

#         # Debug: Test a simple request first
#         print("[@test] Making first test request...")
#         start_time = time.time()

#         try:
#             response = chat(
#                 model='deepseek-r1:70b',  # Explicitly specify the model
#                 messages=[{
#                     "role": "user",
#                     "content": "Hello, can you respond with exactly the text 'TEST SUCCESSFUL'?"
#                 }]
#             )

#             duration = time.time() - start_time
#             print(f"[@test] First request successful in {duration:.1f}s")
#             print(f"[@test] Response: {response}")

#             # Try a few more requests to test consistency
#             for i in range(5):
#                 print(f"[@test] Making request {i+2}...")
#                 req_start = time.time()

#                 response = chat(
#                     model='deepseek-r1:70b',
#                     messages=[{
#                         "role": "user",
#                         "content": f"Request {i+2}: Count to {i+3}"
#                     }]
#                 )

#                 req_duration = time.time() - req_start
#                 print(f"[@test] Request {i+2} completed in {req_duration:.1f}s")

#         except Exception as e:
#             print(f"[@test] REQUEST FAILED: {e}")
#             print(f"[@test] Exception type: {type(e)}")

#         total_duration = time.time() - start_time
#         print(f"[@test] Debug test complete in {total_duration:.1f}s")
#         self.next(self.end)

#     @step
#     def end(self):
#         pass

# if __name__ == '__main__':
#     DeepSeekDebugTest()

from metaflow import FlowSpec, step, ollama, kubernetes, pypi
from metaflow.profilers import gpu_profile


class DeepSeekStressTest(FlowSpec):

    stress_duration = 75  # n_minutes

    @pypi(packages={"ollama": "0.5.1"})
    @kubernetes(
        cpu=32, gpu=4, memory=164_000, shared_memory=128_000, compute_pool="a10g4x"
    )
    @gpu_profile(interval=1)
    # @ollama(models=['deepseek-r1:70b'])
    @ollama(models=["qwen3:0.6b"])
    @step
    def start(self):
        import time
        from ollama import chat

        print(f"[@test] Starting {self.stress_duration}-minute long-running test")

        start_time = time.time()
        end_time = start_time + (self.stress_duration * 60)  # Convert to seconds

        request_count = 0
        successful_requests = 0
        failed_requests = 0
        circuit_breaker_events = []

        while time.time() < end_time:
            try:
                request_start = time.time()
                request_count += 1

                # Vary the complexity of requests
                if request_count % 10 == 0:
                    # Every 10th request is more complex
                    content = f"Request {request_count}: Write a detailed analysis of machine learning trends in 200 words."
                else:
                    content = f"Request {request_count}: Explain one benefit of AI in 30 words."

                response = chat(
                    # model='deepseek-r1:70b',
                    model="qwen3:0.6b",
                    messages=[{"role": "user", "content": content}],
                )

                successful_requests += 1
                duration = time.time() - request_start

                # Log progress every 50 requests
                if request_count % 50 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    print(
                        f"[@test] Progress: {request_count} requests in {elapsed_minutes:.1f} minutes "
                        f"(Success rate: {successful_requests/request_count*100:.1f}%)"
                    )

            except RuntimeError as e:
                if "Circuit Breaker" in str(e):
                    circuit_breaker_events.append(
                        {
                            "timestamp": time.time(),
                            "request_number": request_count,
                            "error": str(e),
                        }
                    )
                    print(
                        f"[@test] Circuit breaker activated at request {request_count}: {e}"
                    )
                    # Wait a bit before retrying
                    time.sleep(5)
                else:
                    failed_requests += 1
                    print(f"[@test] Request {request_count} failed: {e}")

            except Exception as e:
                failed_requests += 1
                print(f"[@test] Request {request_count} failed: {e}")

            # Small delay to prevent overwhelming the system
            time.sleep(1)

        total_duration = time.time() - start_time

        self.long_running_results = {
            "duration_minutes": total_duration / 60,
            "total_requests": request_count,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (
                successful_requests / request_count if request_count > 0 else 0
            ),
            "circuit_breaker_events": len(circuit_breaker_events),
            "circuit_breaker_details": circuit_breaker_events,
            "requests_per_minute": request_count / (total_duration / 60),
        }

        print(f"[@test] Long-running test complete!")
        print(f"[@test] Duration: {total_duration/60:.1f} minutes")
        print(
            f"[@test] Requests: {request_count} ({successful_requests} success, {failed_requests} failed)"
        )
        print(f"[@test] Circuit breaker events: {len(circuit_breaker_events)}")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DeepSeekStressTest()
