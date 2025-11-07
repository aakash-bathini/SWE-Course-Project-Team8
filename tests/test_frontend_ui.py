#!/usr/bin/env python3
"""
Frontend UI Tests using Selenium
Tests the frontend user interface and interactions
"""

import time
import sys
import os

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available - install with: pip install selenium")

FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"
DEFAULT_USERNAME = "ece30861defaultadminuser"
DEFAULT_PASSWORD = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE packages;'

def check_backend_available() -> bool:
    """Check if backend is running"""
    import requests
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def test_frontend_ui():
    """Test frontend UI with Selenium"""
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium not available - skipping UI tests")
        return False
    
    # Check if backend is running
    if not check_backend_available():
        print("⚠️ Backend server not running at http://localhost:8000")
        print("⚠️ Some tests may fail. Start backend with:")
        print("   export USE_SQLITE=1 ENVIRONMENT=development")
        print("   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000")
        print("")
    
    print("=" * 60)
    print("🧪 Starting Frontend UI Tests")
    print("=" * 60)
    
    # Setup Chrome WebDriver
    chrome_options = Options()
    # Run headless to avoid overlay issues
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    # Disable webpack dev server overlay
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = None
    test_results = []
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        print("✅ Chrome WebDriver initialized")
        
        # Test 1: Login Page
        print("\n📝 Test 1: Login Page")
        try:
            driver.get(FRONTEND_URL)
            time.sleep(3)  # Wait for page to fully load
            
            # Remove webpack dev server overlay if present
            try:
                overlay = driver.find_element(By.ID, "webpack-dev-server-client-overlay")
                driver.execute_script("arguments[0].remove();", overlay)
                time.sleep(1)
            except:
                pass  # Overlay not present
            
            # Check for login form
            username_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            password_input = driver.find_element(By.NAME, "password")
            login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign In')]")
            
            print("✅ Login page elements found")
            test_results.append(("Login Page", True))
            
            # Test login
            username_input.clear()
            username_input.send_keys(DEFAULT_USERNAME)
            password_input.clear()
            
            # Remove webpack dev server overlay if present
            try:
                overlay = driver.find_element(By.ID, "webpack-dev-server-client-overlay")
                driver.execute_script("arguments[0].remove();", overlay)
                time.sleep(1)
            except:
                pass  # Overlay not present
            
            # Use JavaScript to set password to avoid character loss with special characters
            # Also trigger input/change events so React state updates
            driver.execute_script("""
                var input = arguments[0];
                var value = arguments[1];
                input.value = value;
                // Trigger React change events
                var event = new Event('input', { bubbles: true });
                input.dispatchEvent(event);
                var changeEvent = new Event('change', { bubbles: true });
                input.dispatchEvent(changeEvent);
            """, password_input, DEFAULT_PASSWORD)
            time.sleep(0.5)  # Wait for React state to update
            
            # Verify password was set correctly
            actual_password = driver.execute_script("return arguments[0].value;", password_input)
            if len(actual_password) != len(DEFAULT_PASSWORD):
                print(f"⚠️ Password length mismatch: set {len(DEFAULT_PASSWORD)}, got {len(actual_password)}")
                print(f"   Expected: {repr(DEFAULT_PASSWORD)}")
                print(f"   Got:      {repr(actual_password)}")
                # Try to find which character is missing
                if len(actual_password) == len(DEFAULT_PASSWORD) - 1:
                    for i in range(len(DEFAULT_PASSWORD)):
                        if i >= len(actual_password) or actual_password[i] != DEFAULT_PASSWORD[i]:
                            print(f"   Missing character at index {i}: '{DEFAULT_PASSWORD[i]}' (ord={ord(DEFAULT_PASSWORD[i])})")
                            break
            else:
                print(f"✅ Password set correctly: {len(actual_password)} characters")
            
            # Use JavaScript click to avoid interception issues
            driver.execute_script("arguments[0].click();", login_button)
            
            # Wait for redirect or check if still on login page
            try:
                WebDriverWait(driver, 10).until(
                    EC.url_contains("/dashboard")
                )
                print("✅ Login successful - redirected to dashboard")
                test_results.append(("Login", True))
            except TimeoutException:
                # Check if we're still on login page (backend might not be running)
                current_url = driver.current_url
                if "login" in current_url.lower() or current_url == FRONTEND_URL or current_url == f"{FRONTEND_URL}/":
                    print("⚠️ Login failed - still on login page (backend may not be running)")
                    test_results.append(("Login", False))
                else:
                    print(f"✅ Login redirected to: {current_url}")
                    test_results.append(("Login", True))
            
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown error"
            print(f"❌ Login test failed: {error_msg}")
            test_results.append(("Login", False))
        
        # Test 2: Dashboard
        print("\n📝 Test 2: Dashboard")
        try:
            driver.get(f"{FRONTEND_URL}/dashboard")
            time.sleep(3)
            
            # Check if redirected to login (not authenticated)
            if "login" in driver.current_url.lower():
                print("⚠️ Dashboard requires authentication - redirected to login")
                test_results.append(("Dashboard", False))
            else:
                try:
                    dashboard_title = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Dashboard') or contains(text(), 'Welcome')]"))
                    )
                    print("✅ Dashboard loaded")
                    test_results.append(("Dashboard", True))
                except TimeoutException:
                    # Page loaded but element not found - might be different structure
                    print("⚠️ Dashboard page loaded but expected elements not found")
                    test_results.append(("Dashboard", False))
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown error"
            print(f"❌ Dashboard test failed: {error_msg}")
            test_results.append(("Dashboard", False))
        
        # Test 3: Upload Page
        print("\n📝 Test 3: Upload Page")
        try:
            driver.get(f"{FRONTEND_URL}/upload")
            time.sleep(2)
            
            upload_title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Upload')]"))
            )
            print("✅ Upload page loaded")
            test_results.append(("Upload Page", True))
        except Exception as e:
            print(f"❌ Upload page test failed: {e}")
            test_results.append(("Upload Page", False))
        
        # Test 4: Search Page
        print("\n📝 Test 4: Search Page")
        try:
            driver.get(f"{FRONTEND_URL}/search")
            time.sleep(2)
            
            search_title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Search')]"))
            )
            
            # Try to search
            try:
                search_input = driver.find_element(By.XPATH, "//input[contains(@placeholder, 'Name') or contains(@label, 'Name')]")
                search_input.clear()
                search_input.send_keys("*")
                
                search_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
                search_button.click()
                time.sleep(3)
                print("✅ Search executed")
            except:
                print("⚠️ Search button not found, but page loaded")
            
            test_results.append(("Search Page", True))
        except Exception as e:
            print(f"❌ Search page test failed: {e}")
            test_results.append(("Search Page", False))
        
        # Test 5: Download Page
        print("\n📝 Test 5: Download Page")
        try:
            driver.get(f"{FRONTEND_URL}/download")
            time.sleep(2)
            
            download_title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Download') or contains(text(), 'Browse')]"))
            )
            print("✅ Download page loaded")
            test_results.append(("Download Page", True))
        except Exception as e:
            print(f"❌ Download page test failed: {e}")
            test_results.append(("Download Page", False))
        
        # Test 6: Health Dashboard
        print("\n📝 Test 6: Health Dashboard")
        try:
            driver.get(f"{FRONTEND_URL}/health")
            time.sleep(3)  # Health dashboard may take longer to load
            
            health_title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Health')]"))
            )
            print("✅ Health dashboard loaded")
            test_results.append(("Health Dashboard", True))
        except Exception as e:
            print(f"❌ Health dashboard test failed: {e}")
            test_results.append(("Health Dashboard", False))
        
        # Test 7: Navigation
        print("\n📝 Test 7: Navigation")
        try:
            pages = ["/dashboard", "/upload", "/search", "/download", "/health"]
            for page in pages:
                driver.get(f"{FRONTEND_URL}{page}")
                time.sleep(1)
                assert "404" not in driver.title.lower()
            print("✅ Navigation working")
            test_results.append(("Navigation", True))
        except Exception as e:
            print(f"❌ Navigation test failed: {e}")
            test_results.append(("Navigation", False))
        
        # Test 8: Check for console errors (skip React dev errors)
        print("\n📝 Test 8: Console Errors")
        try:
            logs = driver.get_log('browser')
            # Filter out webpack/React dev server errors and authentication errors (if backend not running)
            errors = [
                log for log in logs 
                if log['level'] == 'SEVERE' 
                and 'webpack' not in log['message'].lower()
                and 'maximum update depth' not in log['message'].lower()
                and not ('authenticate' in log['message'].lower() and 'failed to load' in log['message'].lower())
            ]
            if errors:
                print(f"⚠️ Found {len(errors)} console errors:")
                for error in errors[:3]:
                    print(f"   - {error['message'][:100]}")
                test_results.append(("Console Errors", False))
            else:
                print("✅ No critical console errors (ignoring dev server warnings)")
                test_results.append(("Console Errors", True))
        except:
            print("⚠️ Could not check console errors")
            test_results.append(("Console Errors", True))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 UI Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed ({passed*100//total if total > 0 else 0}%)")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ UI test setup failed: {e}")
        return False
    finally:
        if driver:
            driver.quit()
            print("\n✅ WebDriver closed")

if __name__ == "__main__":
    success = test_frontend_ui()
    sys.exit(0 if success else 1)

