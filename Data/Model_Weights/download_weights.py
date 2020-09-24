# -----------------------------
#   USAGE
# -----------------------------
# python download_weights.py drive_file_id destination_file_path

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import requests
import os
import progressbar
import sys


# -----------------------------
#   FUNCTIONS
# -----------------------------
def download_file_from_google_drive(ID, destination):
    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(resp, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as f:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            i = 0
            for chunk in resp.iter_content(CHUNK_SIZE):
                # Filter out keep-alive new chunks
                if chunk:
                    bar.update(i)
                    i += 1
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": ID}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": ID, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


# -----------------------------
#   MAIN
# -----------------------------
if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print("Usage: python download_weights.py drive_file_id destination_file_path")
    else:
        # Take ID from the shareable link
        file_id = sys.argv[1]
        # Save destination file on your disk
        destination = os.path.join(os.getcwd(), sys.argv[2])
        download_file_from_google_drive(file_id, destination)
