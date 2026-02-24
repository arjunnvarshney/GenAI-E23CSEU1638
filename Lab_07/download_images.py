import requests
import os

def download_image(url, filename):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}, Status: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

content_url = "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg"
# Style: Starry Night
style_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

os.makedirs("Lab_07/inputs", exist_ok=True)
download_image(content_url, "Lab_07/inputs/content.jpg")
download_image(style_url, "Lab_07/inputs/style.jpg")
