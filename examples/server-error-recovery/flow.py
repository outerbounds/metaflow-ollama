from metaflow import FlowSpec, step, pypi, ollama, kubernetes, Parameter, current, card
from metaflow.profilers import gpu_profile
from metaflow.cards import ProgressBar
import time


class ProductionOllamaHandlingFlow(FlowSpec):
    """
    Essential production patterns for Ollama: error handling and exponential retry.

    This example demonstrates the core concepts every user needs:
    - Proper RuntimeError handling for circuit breaker states
    - Exponential backoff retry logic
    - Circuit breaker configuration
    - Smart exit conditions
    """

    max_requests = Parameter(
        "max-requests",
        help="Maximum number of requests to process",
        type=int,
        default=1000,
    )

    @card(id="progress")
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=16000, cpu=4)
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["llama3.2:3b"],
        # debug=True,
        # Essential circuit breaker configuration
        circuit_breaker_config={
            "failure_threshold": 3,  # Open after 3 failures
            "recovery_timeout": 30,  # Test recovery after 30s
            "reset_timeout": 60,  # Restart after 60s open
        },
    )
    @step
    def start(self):
        """
        Demonstrates robust request processing with exponential backoff retry.

        Key patterns shown:
        1. Handle RuntimeError exceptions (circuit breaker)
        2. Implement exponential backoff for retries
        3. Track metrics for operational visibility
        4. Use smart exit conditions to prevent waste
        """
        print("[@prod] Starting production error handling demo")

        # Track basic metrics
        successful_requests = 0
        failed_requests = 0
        circuit_breaker_hits = 0

        # Process requests with robust error handling
        p = ProgressBar(max=self.max_requests, label="Requests processed")
        current.card["progress"].append(p)
        current.card["progress"].refresh()

        for request_id in range(self.max_requests):
            print(f"[@prod] Processing request {request_id + 1}/{self.max_requests}")

            success = self._robust_request_with_retry(
                request_id, max_retries=10, base_wait_time=2
            )

            if success:
                successful_requests += 1
                p.update(request_id)
                current.card["progress"].refresh()
                print(f"[@prod] ✓ Request {request_id} succeeded")
            else:
                failed_requests += 1
                print(f"[@prod] ✗ Request {request_id} failed after retries")

                # Check if it was a circuit breaker failure
                if self._last_error_was_circuit_breaker:
                    circuit_breaker_hits += 1

            # Smart exit conditions - save compute when things go wrong
            total_requests = successful_requests + failed_requests
            if total_requests >= 10:
                success_rate = successful_requests / total_requests

                # Exit early if success rate is too low
                if success_rate < 0.3:
                    print(
                        f"[@prod] Early exit: Success rate too low ({success_rate:.1%})"
                    )
                    break

                # Exit if circuit breaker is getting hit too often
                if circuit_breaker_hits >= 3:
                    print(
                        f"[@prod] Early exit: Circuit breaker hit {circuit_breaker_hits} times"
                    )
                    break

            # Progress updates
            if (request_id + 1) % 10 == 0:
                success_rate = (
                    successful_requests / total_requests if total_requests > 0 else 0
                )
                print(
                    f"[@prod] Progress: {success_rate:.1%} success rate, {circuit_breaker_hits} circuit breaker hits"
                )

        # Store results for end step
        self.results = {
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "circuit_breaker_hits": circuit_breaker_hits,
            "total_requests": successful_requests + failed_requests,
        }

        print(
            f"[@prod] Processing complete: {successful_requests} successful, {failed_requests} failed"
        )
        self.next(self.end)

    def _robust_request_with_retry(
        self, request_id: int, max_retries: int = 10, base_wait_time: int = 2
    ) -> bool:
        """
        THE CORE PATTERN: Robust request with exponential backoff retry.

        This is the essential function every production Ollama user needs.

        max_retries is the important parameter here, which needs to play in sync with how the circuit breaker in @ollama resets the server.
        The key insight is to make this number big enough so this function doesn't give up retrying as the circuit breaker is resetting the server,
        then this request gets unnecessarily dropped when it could have just waited a bit more.
        """
        from ollama import chat

        self._last_error_was_circuit_breaker = False

        for attempt in range(max_retries):
            try:
                # Make the actual request
                response = chat(
                    model="llama3.2:3b",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Request {request_id}: What is cloud computing? Answer in 50 words.",
                        }
                    ],
                    options={"num_predict": 100},  # Limit response length
                )

                # Success!
                return True

            except RuntimeError as e:
                # This is the KEY exception to handle - circuit breaker errors
                error_msg = str(e)

                if "Circuit Breaker" in error_msg:
                    self._last_error_was_circuit_breaker = True
                    print(
                        f"[@prod] Request {request_id}, attempt {attempt + 1}: Circuit breaker open"
                    )

                    # Exponential backoff: 2s, 4s, 8s, etc. (capped at 30s)
                    wait_time = min(base_wait_time * (2**attempt), 30)
                    print(f"[@prod] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

                    # Continue to next retry
                    continue
                else:
                    # Other runtime error - don't retry
                    print(
                        f"[@prod] Request {request_id}: Non-recoverable runtime error: {error_msg}"
                    )
                    return False

            except Exception as e:
                # Unexpected error - log and retry with exponential backoff
                print(
                    f"[@prod] Request {request_id}, attempt {attempt + 1}: {type(e).__name__}: {e}"
                )

                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    wait_time = min(base_wait_time * (2**attempt), 10)
                    print(f"[@prod] Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries exhausted
        print(f"[@prod] Request {request_id}: Failed after {max_retries} attempts")
        return False

    @step
    def end(self):
        """
        Summary and key takeaways for production use.
        """
        results = self.results
        total = results["total_requests"]
        success_rate = results["successful_requests"] / total if total > 0 else 0

        print("[@prod] ")
        print("[@prod] 📊 FINAL RESULTS:")
        print(f"[@prod]   Total requests: {total}")
        print(f"[@prod]   Successful: {results['successful_requests']}")
        print(f"[@prod]   Failed: {results['failed_requests']}")
        print(f"[@prod]   Success rate: {success_rate:.1%}")
        print(f"[@prod]   Circuit breaker hits: {results['circuit_breaker_hits']}")
        print("[@prod] ")
        print("[@prod] IMPORTANT PRODUCTION PATTERNS:")
        print("[@prod]   1. Always catch RuntimeError for circuit breaker states")
        print("[@prod]   2. Implement exponential backoff: 2s, 4s, 8s, 16s...")
        print("[@prod]   3. Cap maximum wait time (e.g., 30 seconds).")
        print("[@prod]   4. Use smart exit conditions to save compute costs.")
        print(
            "[@prod]   5. Track circuit breaker hits for monitoring (Automatic in ollama_status card.)"
        )
        print("[@prod] ")

        if success_rate > 0.95:
            print("[@prod] ✅ Excellent performance! Your system is production-ready.")
        elif success_rate > 0.8:
            print("[@prod] ⚠️  Good performance, but monitor circuit breaker behavior.")
        else:
            print(
                "[@prod] 🚨 Low success rate - investigate system issues before production."
            )


if __name__ == "__main__":
    ProductionOllamaHandlingFlow()
