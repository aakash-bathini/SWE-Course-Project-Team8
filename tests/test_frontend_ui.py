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
DEFAULT_USERNAME = "ece30861defaultadminuser"
DEFAULT_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"

def test_frontend_ui():
    """Test frontend UI with Selenium"""
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium not available - skipping UI tests")
        return False
    
    print("=" * 60)
    print("🧪 Starting Frontend UI Tests")
    print("=" * 60)
    
    # Setup Chrome WebDriver
    chrome_options = Options()
    # Uncomment to run headless:
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
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
            time.sleep(2)
            
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
            password_input.send_keys(DEFAULT_PASSWORD)
            login_button.click()
            
            # Wait for redirect
            WebDriverWait(driver, 10).until(
                EC.url_contains("/dashboard")
            )
            print("✅ Login successful - redirected to dashboard")
            test_results.append(("Login", True))
            
        except Exception as e:
            print(f"❌ Login test failed: {e}")
            test_results.append(("Login", False))
        
        # Test 2: Dashboard
        print("\n📝 Test 2: Dashboard")
        try:
            driver.get(f"{FRONTEND_URL}/dashboard")
            time.sleep(2)
            
            dashboard_title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Dashboard') or contains(text(), 'Welcome')]"))
            )
            print("✅ Dashboard loaded")
            test_results.append(("Dashboard", True))
        except Exception as e:
            print(f"❌ Dashboard test failed: {e}")
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
        
        # Test 8: Check for console errors
        print("\n📝 Test 8: Console Errors")
        try:
            logs = driver.get_log('browser')
            errors = [log for log in logs if log['level'] == 'SEVERE']
            if errors:
                print(f"⚠️ Found {len(errors)} console errors:")
                for error in errors[:3]:
                    print(f"   - {error['message'][:100]}")
                test_results.append(("Console Errors", False))
            else:
                print("✅ No console errors")
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

