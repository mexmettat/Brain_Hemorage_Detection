import os
import requests

def download_image(url, folder, filename):
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            file_path = os.path.join(folder, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded: {filename}")
        else:
            print(f"Failed to download {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def collect_presentation_samples():
    output_dir = "data/presentation_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        {
            "type": "epidural",
            "url": "https://prod-images-static.radiopaedia.org/images/5286882/368ab7c53ae94eb1b2146062a26443.jpeg",
            "filename": "epidural_sample.jpg"
        },
        {
            "type": "subdural",
            "url": "https://prod-images-static.radiopaedia.org/images/5292072/aa03e8e5db64f30b85180f3fa4215c.jpeg",
            "filename": "subdural_sample.jpg"
        },
        {
            "type": "subarachnoid",
            "url": "https://prod-images-static.radiopaedia.org/images/529732/475f2192d88b21ae8f29b62d207910.jpg",
            "filename": "subarachnoid_sample.jpg"
        },
        {
            "type": "intraparenchymal",
            "url": "https://prod-images-static.radiopaedia.org/images/4605130/dd1e4a3390c07d0d7cb2018a78e5d1.jpg",
            "filename": "intraparenchymal_sample.jpg"
        },
        {
            "type": "intraventricular",
            "url": "https://prod-images-static.radiopaedia.org/images/30931235/115a8e0bd61e2eb1304b1d4b997f1e.jpg",
            "filename": "intraventricular_sample.jpg"
        },
        {
            "type": "normal",
            "url": "https://prod-images-static.radiopaedia.org/images/58335750/a60492e46b37a152fe2660a6090824a88f1d635fffc80836507b303efdb14280.jpeg",
            "filename": "normal_sample_1.jpg"
        },
        {
            "type": "normal",
            "url": "https://prod-images-static.radiopaedia.org/images/4170243/a0bfe454334c860c146609d72c562c.jpg",
            "filename": "normal_sample_2.jpg"
        },
        {
            "type": "normal",
            "url": "https://prod-images-static.radiopaedia.org/images/14863208/aed4f3092b7f45559582feb89487e8.jpg",
            "filename": "normal_sample_3.jpg"
        }
    ]
    
    print(f"Starting collection of {len(samples)} external samples...")
    for sample in samples:
        download_image(sample['url'], output_dir, sample['filename'])
    print("\nCollection complete! Files saved in: data/presentation_samples/")

if __name__ == "__main__":
    collect_presentation_samples()
