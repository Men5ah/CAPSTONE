import requests
import concurrent.futures

# Install requests[socks] if not already installed
try:
    import socks
except ImportError:
    print("Missing dependency! Run: pip install requests[socks]")
    exit()

# File paths
PROXY_FILE = "Bot/proxies/data.txt"  # Replace with your proxy list file
OUTPUT_FILE = "working_proxies.txt"
TEST_URL = "https://httpbin.org/ip"  # Checks if the proxy works

# Load proxies from a file
def load_proxies(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]

# Detect protocol type based on port number or prefix
def detect_proxy_type(proxy):
    if proxy.startswith("socks5://"):
        return "socks5"
    elif proxy.startswith("socks4://"):
        return "socks4"
    elif ":1080" in proxy or ":9050" in proxy:  # Common SOCKS5 ports
        return "socks5"
    elif proxy.lower().startswith("https"):
        return "https"
    else:
        return "http"

# Test a single proxy
def test_proxy(proxy):
    proxy_type = detect_proxy_type(proxy)
    proxy_dict = {proxy_type: f"{proxy_type}://{proxy}"}

    try:
        response = requests.get(TEST_URL, proxies=proxy_dict, timeout=5)
        if response.status_code == 200:
            print(f"✅ WORKING: {proxy} | IP: {response.json()}")
            return proxy  # Return only working proxies
    except requests.exceptions.RequestException:
        print(f"❌ FAILED: {proxy}")
    return None

# Main function to test proxies in parallel
def main():
    proxies = load_proxies(PROXY_FILE)
    working_proxies = []

    print(f"🔍 Testing {len(proxies)} proxies...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(test_proxy, proxies)

    # Filter out None values (failed proxies)
    working_proxies = [proxy for proxy in results if proxy]

    # Save working proxies to a file
    with open(OUTPUT_FILE, "w") as file:
        file.write("\n".join(working_proxies))

    print(f"\n✅ {len(working_proxies)} working proxies saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
