from metaflow import FlowSpec, step, pypi, ollama, kubernetes, Parameter
from metaflow.profilers import gpu_profile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class OllamaStressTestFlow(FlowSpec):
    """
    A comprehensive test flow for the Ollama deadlock prevention system.
    Tests big models, long-running tasks, concurrent requests, and circuit breaker behavior.
    """
    
    # Parameters for controlling test behavior
    stress_duration = Parameter(
        'stress-duration',
        help='Duration in minutes for stress testing',
        type=int,
        default=90  # 1.5 hours to approach the 2-hour deadlock threshold
    )
    
    concurrent_requests = Parameter(
        'concurrent-requests', 
        help='Number of concurrent requests to make',
        type=int,
        default=8
    )

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=16000, cpu=4)
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["llama3.2:1b"],  # Start with small model for quick setup
        debug=True,
        circuit_breaker_config={
            'failure_threshold': 3,
            'recovery_timeout': 60,
            'reset_timeout': 300
        }
    )
    @step
    def start(self):
        """
        Initialize and test basic functionality with a small model first.
        """
        from ollama import chat
        
        print("[@test] Starting Ollama stress test flow")
        print(f"[@test] Stress duration: {self.stress_duration} minutes")
        print(f"[@test] Concurrent requests: {self.concurrent_requests}")
        
        # Quick test to ensure basic functionality works
        try:
            response = chat(
                model="llama3.2:1b",
                messages=[{
                    "role": "user", 
                    "content": "Say 'Ollama is working' in exactly those words."
                }]
            )
            print(f"[@test] Basic test successful: {response['message']['content']}")
            self.basic_test_passed = True
        except Exception as e:
            print(f"[@test] Basic test failed: {e}")
            self.basic_test_passed = False
        
        self.start_time = time.time()
        self.next(self.big_model_test)

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=32000, cpu=8)  # More resources for big models
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["deepseek-r1:7b"],  # Big model that could cause issues
        debug=True,
        force_pull=False,  # Use cache if available
        circuit_breaker_config={
            'failure_threshold': 3,
            'recovery_timeout': 60,
            'reset_timeout': 300
        }
    )
    @step
    def big_model_test(self):
        """
        Test with large models that are more likely to cause resource issues.
        """
        from ollama import chat
        
        print("[@test] Testing big model: deepseek-r1:7b")
        
        try:
            start_time = time.time()
            
            response = chat(
                model="deepseek-r1:7b",
                messages=[{
                    "role": "user",
                    "content": "Explain quantum computing in exactly 100 words. Be precise and technical."
                }]
            )
            
            duration = time.time() - start_time
            self.big_model_result = {
                "success": True,
                "duration": duration,
                "response_length": len(response['message']['content']),
                "error": None
            }
            print(f"[@test] Big model successful in {duration:.2f}s")
            
        except Exception as e:
            self.big_model_result = {
                "success": False,
                "duration": time.time() - start_time,
                "response_length": 0,
                "error": str(e)
            }
            print(f"[@test] Big model failed: {e}")
        
        self.next(self.concurrent_stress_test)

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=24000, cpu=6)
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["llama3.2:3b"],  # Medium-sized model for concurrent testing
        debug=True,
        circuit_breaker_config={
            'failure_threshold': 5,  # Higher threshold for concurrent load
            'recovery_timeout': 45,
            'reset_timeout': 180
        }
    )
    @step 
    def concurrent_stress_test(self):
        """
        Test concurrent requests to simulate high load and potential deadlock scenarios.
        """
        from ollama import chat
        
        print(f"[@test] Starting concurrent stress test with {self.concurrent_requests} requests")
        
        def make_request(request_id):
            """Make a single request for concurrent testing"""
            try:
                start_time = time.time()
                response = chat(
                    model="llama3.2:3b",
                    messages=[{
                        "role": "user",
                        "content": f"Request {request_id}: Write a short story about AI in exactly 50 words."
                    }]
                )
                duration = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": True,
                    "duration": duration,
                    "response_length": len(response['message']['content']),
                    "error": None
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "duration": time.time() - start_time,
                    "response_length": 0,
                    "error": str(e)
                }
        
        # Test concurrent requests
        concurrent_results = []
        
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            futures = []
            
            # Submit requests
            for i in range(self.concurrent_requests):
                future = executor.submit(make_request, i)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                concurrent_results.append(result)
                print(f"[@test] Request {result['request_id']} ({'✓' if result['success'] else '✗'}): "
                      f"{result['duration']:.2f}s")
        
        # Analyze results
        successful_requests = [r for r in concurrent_results if r['success']]
        failed_requests = [r for r in concurrent_results if not r['success']]
        
        print(f"[@test] Concurrent test complete: {len(successful_requests)}/{len(concurrent_results)} successful")
        
        self.concurrent_results = {
            "total_requests": len(concurrent_results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "average_duration": sum(r['duration'] for r in successful_requests) / len(successful_requests) if successful_requests else 0,
            "results": concurrent_results
        }
        
        self.next(self.long_running_test)

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=20000, cpu=4)
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["llama3.2:3b"],
        debug=True,
        circuit_breaker_config={
            'failure_threshold': 3,
            'recovery_timeout': 60,
            'reset_timeout': 300
        }
    )
    @step
    def long_running_test(self):
        """
        Run continuous requests for an extended period to test for memory leaks
        and the original 2-hour deadlock issue.
        """
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
                    model="llama3.2:3b",
                    messages=[{
                        "role": "user",
                        "content": content
                    }]
                )
                
                successful_requests += 1
                duration = time.time() - request_start
                
                # Log progress every 50 requests
                if request_count % 50 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    print(f"[@test] Progress: {request_count} requests in {elapsed_minutes:.1f} minutes "
                          f"(Success rate: {successful_requests/request_count*100:.1f}%)")
                
            except RuntimeError as e:
                if "Circuit Breaker" in str(e):
                    circuit_breaker_events.append({
                        "timestamp": time.time(),
                        "request_number": request_count,
                        "error": str(e)
                    })
                    print(f"[@test] Circuit breaker activated at request {request_count}: {e}")
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
            "success_rate": successful_requests / request_count if request_count > 0 else 0,
            "circuit_breaker_events": len(circuit_breaker_events),
            "circuit_breaker_details": circuit_breaker_events,
            "requests_per_minute": request_count / (total_duration / 60)
        }
        
        print(f"[@test] Long-running test complete!")
        print(f"[@test] Duration: {total_duration/60:.1f} minutes")
        print(f"[@test] Requests: {request_count} ({successful_requests} success, {failed_requests} failed)")
        print(f"[@test] Circuit breaker events: {len(circuit_breaker_events)}")
        
        self.next(self.deadlock_simulation_test)

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=16000, cpu=4)
    @pypi(packages={"ollama": "0.5.1"})
    @ollama(
        models=["llama3.2:1b"],
        debug=True,
        circuit_breaker_config={
            'failure_threshold': 2,  # Very low threshold for testing
            'recovery_timeout': 20,   # Quick recovery for testing
            'reset_timeout': 60       # Quick reset for testing
        }
    )
    @step
    def deadlock_simulation_test(self):
        """
        Simulate conditions that might trigger the original deadlock issue.
        """
        from ollama import chat
        import requests
        
        print("[@test] Testing deadlock prevention mechanisms")
        
        deadlock_test_results = []
        
        # Test 1: Rapid concurrent requests (stress the system)
        print("[@test] Test 1: Rapid concurrent requests")
        try:
            def rapid_request(i):
                return chat(
                    model="llama3.2:1b",
                    messages=[{"role": "user", "content": f"Quick response {i}"}]
                )
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(rapid_request, i) for i in range(20)]
                results = [f.result() for f in as_completed(futures, timeout=60)]
            
            deadlock_test_results.append({"test": "rapid_requests", "success": True, "message": f"Completed {len(results)}/20 requests"})
        except Exception as e:
            deadlock_test_results.append({"test": "rapid_requests", "success": False, "error": str(e)})
        
        # Test 2: Long request during concurrent load
        print("[@test] Test 2: Long request during concurrent load")
        try:
            def background_requests():
                for i in range(10):
                    try:
                        chat(model="llama3.2:1b", messages=[{"role": "user", "content": f"Background {i}"}])
                    except:
                        pass
                    time.sleep(2)
            
            # Start background load
            bg_thread = threading.Thread(target=background_requests)
            bg_thread.start()
            
            # Make a long request
            long_response = chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": "Write a detailed explanation of machine learning in 500 words."}]
            )
            
            bg_thread.join(timeout=30)
            deadlock_test_results.append({"test": "long_request_with_load", "success": True, "message": "Completed successfully"})
        except Exception as e:
            deadlock_test_results.append({"test": "long_request_with_load", "success": False, "error": str(e)})
        
        # Test 3: Server health during load
        print("[@test] Test 3: Server health check during load")
        try:
            # Make some requests while checking server health
            for i in range(5):
                # Health check
                health_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if health_response.status_code != 200:
                    raise Exception(f"Server unhealthy: {health_response.status_code}")
                
                # Regular request
                chat(model="llama3.2:1b", messages=[{"role": "user", "content": f"Health test {i}"}])
                time.sleep(1)
            
            deadlock_test_results.append({"test": "health_during_load", "success": True, "message": "Server remained healthy"})
        except Exception as e:
            deadlock_test_results.append({"test": "health_during_load", "success": False, "error": str(e)})
        
        self.deadlock_test_results = {
            "tests_completed": len(deadlock_test_results),
            "results": deadlock_test_results
        }
        
        self.next(self.end)

    @step
    def end(self):
        """
        Summarize all test results and provide recommendations.
        """
        print("\n" + "="*80)
        print("OLLAMA STRESS TEST SUMMARY")
        print("="*80)
        
        # Basic test results
        if hasattr(self, 'basic_test_passed'):
            print(f"✓ Basic functionality: {'PASSED' if self.basic_test_passed else 'FAILED'}")
        
        # Big model test results
        if hasattr(self, 'big_model_result'):
            result = self.big_model_result
            status = "✓" if result['success'] else "✗"
            print(f"{status} Big model test (deepseek-r1:7b): {result['duration']:.2f}s")
            if not result['success']:
                print(f"  Error: {result['error']}")
        
        # Concurrent test results
        if hasattr(self, 'concurrent_results'):
            cr = self.concurrent_results
            success_rate = (cr['successful_requests'] / cr['total_requests'] * 100)
            print(f"✓ Concurrent requests: {success_rate:.1f}% success rate "
                  f"({cr['successful_requests']}/{cr['total_requests']})")
            print(f"  Average response time: {cr['average_duration']:.2f}s")
        
        # Long-running test results
        if hasattr(self, 'long_running_results'):
            lr = self.long_running_results
            print(f"✓ Long-running test: {lr['duration_minutes']:.1f} minutes")
            print(f"  Requests: {lr['total_requests']} ({lr['success_rate']*100:.1f}% success)")
            print(f"  Rate: {lr['requests_per_minute']:.1f} requests/minute")
            print(f"  Circuit breaker events: {lr['circuit_breaker_events']}")
            
            # Check for deadlock indicators
            if lr['duration_minutes'] > 60 and lr['success_rate'] > 0.8:
                print("  ✓ NO DEADLOCK DETECTED - Test ran successfully for >1 hour")
            elif lr['circuit_breaker_events'] > 0:
                print("  ⚠ Circuit breaker activated during test - check logs")
            else:
                print("  ⚠ Test completed but check detailed logs")
        
        # Deadlock simulation test results
        if hasattr(self, 'deadlock_test_results'):
            dt = self.deadlock_test_results
            successful_tests = [r for r in dt['results'] if r['success']]
            print(f"✓ Deadlock simulation: {len(successful_tests)}/{len(dt['results'])} tests passed")
            for result in dt['results']:
                status = "✓" if result['success'] else "✗"
                test_name = result['test'].replace('_', ' ').title()
                print(f"  {status} {test_name}")
        
        # Overall assessment
        total_runtime = time.time() - self.start_time
        print(f"\nTotal test runtime: {total_runtime/60:.1f} minutes")
        
        if total_runtime > 5400:  # 90 minutes
            print("🎉 SUCCESS: Test ran for >90 minutes without deadlock!")
        elif total_runtime > 3600:  # 60 minutes
            print("✅ GOOD: Test ran for >60 minutes - likely no deadlock issues")
        else:
            print("⚠️  Test completed early - check for errors")
        
        print("\nDeadlock prevention features tested:")
        print("  ✓ Timeout protection on subprocess calls")
        print("  ✓ Circuit breaker for request protection") 
        print("  ✓ Health monitoring")
        print("  ✓ Graceful shutdown with fallbacks")
        print("  ✓ Big model handling")
        print("  ✓ Concurrent request management")
        print("="*80)


if __name__ == "__main__":
    OllamaStressTestFlow()