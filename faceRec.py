import json
import signal
import sys
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from fast_mtcnn import FastMTCNN
import torch
import glob
import os
from torchvision import transforms, datasets
from pythonosc import udp_client, osc_server, dispatcher
import NDIlib as ndi
import threading
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import requests
import curses

## init curses
#win = curses.initscr()
#win.clear()
#client_win = win.subwin(16, curses.COLS-1, 0, 0)
#log_win = win.subwin(18, 0)
#log_win.scrollok(True)
#win.hline(17,0,"-",curses.COLS)
#win.refresh()
def draw_title(win, title):
    # Clear any existing title
    win.clear()
    
    # Get window dimensions
    height, width = win.getmaxyx()
    
    # Create a sub-window for the title
    title_win = win.subwin(3, width - 2, 0, 1)  # Height: 3, Width: full screen - 2, Position: (0,1)
    
    # Initialize color for title background
    title_win.bkgd(curses.color_pair(1))  # Set background color (color pair 1)
    
    # Draw a box around the title window
    title_win.box()
    
    # Add the title text centered in the box
    title_text = title.center(width - 2)  # Center the title based on the window width
    title_win.addstr(1, 1, title_text)  # Draw title at (1, 1) inside the title window
    
    # Refresh the title window to show changes
    title_win.refresh()

def display_interface(client_win, log_win, faces, current_detected_names, status, cv2_device, torch_device):
    # Clear windows
    client_win.clear()
    log_win.clear()

    # Display each line with a different color
    client_win.addstr(0, 0, f"Known faces in frame: {current_detected_names}", curses.color_pair(1))  # Green
    client_win.addstr(1, 0, f"Detected Faces: {len(faces)}", curses.color_pair(2))                   # Blue
    client_win.addstr(2, 0, f"Status: {status}", curses.color_pair(3))                               # Red
    client_win.addstr(3, 0, f"OpenCV Device: {cv2_device}", curses.color_pair(4))                    # Yellow
    client_win.addstr(4, 0, f"Torch Device: {torch_device}", curses.color_pair(5))                   # Cyan

    # Refresh windows
    client_win.refresh()
    log_win.refresh()

# Initialize curses and create windows
win = curses.initscr()
curses.start_color()
curses.curs_set(0)
win.clear()

curses.start_color()
curses.curs_set(0)  # Hide cursor
curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green
curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)   # Blue
curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Red
curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Yellow
curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Cyan

# Initial window setup
height, width = win.getmaxyx()
client_win = win.subwin(min(16, height - 3), width - 1, 0, 0)
log_win = win.subwin(height - 17, width - 1, 17, 0)
log_win.scrollok(True)
win.hline(16, 0, "-", width)

draw_title(win, "Face Recognition System")

# Load and parse the JSON configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

# Assign the values from JSON config to Python variables
cv_show_render = config['cv_show_render']
input_width = config['input_width']
input_height = config['input_height']
output_width = config['output_width']
output_height = config['output_height']
dataset_folder = config['dataset_folder']
camera_id = config['camera_id']

#SPECIFIFIC USECASES FUNCTIONS================================

# Function to check camera ID
def checkCamID(cen_x, cen_y):
    global ndi_width, ndi_height
    if cen_x < ndi_width/2 and cen_y < ndi_height/2:
        return 1
    elif cen_x > ndi_width/2 and cen_y < ndi_height/2:
        return 2
    elif cen_x < ndi_width/2 and cen_y > ndi_height/2:
        return 3
    elif cen_x > ndi_width/2 and cen_y > ndi_height/2:
        return 4

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Signal received, shutting down...")
    # Clean up NDI resources
    cleanup_ndi()
    # Release the camera
    cap.release()
    # Close any OpenCV windows
    cv2.destroyAllWindows()
    # Shut down the OSC server
    server.shutdown()
    server.server_close()
    # Exit the program
    sys.exit(0)

# Register signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

# NDI Cleanup Function
def cleanup_ndi():
    global ndi_send
    if ndi_send:
        ndi.send_destroy(ndi_send)
        ndi_send = None

    
# Function to send logs to DC webhook

#webhook = REDACTED

#def send_msg(webhook_url, message):
#    payload = {'content': message}
#    headers = {'Content-Type': 'application/json'}
#
#    response = requests.post(webhook_url, json=payload, headers=headers)
#
#    if response.status_code == 204:
#        print('Message sent')
#    else:
#        print(f'{response.status_code}')

# UI CLASSES AND FUNCTIONS ========================================
os.system('color')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display_log(known_faces, detected_faces_count, status, opencv_device, torch_device):
    """Displays dynamic log information with colors, overwriting the previous content."""
    # ANSI color codes
    green = "\033[92m"
    yellow = "\033[93m"
    pink = "\033[95m"
    dark_red = "\033[31m"
    bright_green = "\033[92;1m"
    bold_underline = "\033[1;4m"
    reset = "\033[0m"
    # Move cursor up six lines to overwrite the existing content
    print("\033[6A", end="\r")  # Move up six lines to include the title
    print("\033[K" + bold_underline + "FaceRec" + reset, end="\r\n")  # Clear line and print title
    print("\033[K" + green + "Known faces in frame: " + reset + ', '.join(known_faces), end="\r\n")  # Clear line and print
    print("\033[K" + yellow + "Detected faces in frame: " + reset + str(detected_faces_count), end="\r\n")  # Clear line and print
    print("\033[K" + pink + "Status: " + reset + status, end="\r\n")  # Clear line and print
    print("\033[K" + dark_red + "OpenCV device: " + reset + opencv_device, end="\r\n")  # Clear line and print
    print("\033[K" + bright_green + "Torch device: " + reset + torch_device, end="\r\n")  # Clear line and print

# SERVERS INITIATION ==============================================

# OSC SERVER

# Define handler
def update_db_handler(address, *args):
    # Check if the address is for the updateDB channel
    if address == "/updateDB":
        # Check if the value received is 1
        if len(args) > 0 and args[0] == 1:
            # Count the number of folders in dataset
            update_db(embedding_list, name_list)

# Create a dispatcher
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/updateDB", update_db_handler)

# Set up the OSC server
server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 666), dispatcher)

def run_server():
    server.serve_forever()

# Thread and start server
server_thread = threading.Thread(target=run_server)
server_thread.start()

# OSC CLIENT

osc_host = "127.0.0.1"
osc_port = 10000
osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)

# NDI

send_settings = ndi.SendCreate()
send_settings.ndi_name = 'FaceRecBoxed'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()
ndi_name = b"FACEREC"
ndi_width = output_width
ndi_height = output_height



# CUDA SUPPORT CHECK FOR PYTORCH AND OPENCV ==========================
cv2_cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
cv2_device = "cuda" if cv2_cuda_enabled else "cpu"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# RESNET AND MTCNN INITIATION ========================================

# A faster implementation of MTCNN is used for real-time face detection
fast_mtcnn = FastMTCNN(stride=2, resize=1, margin=0, factor=0.8, keep_all=True, device=torch_device, min_face_size=120)

# Normal MTCNN is used for accurate embeddings during data treatment
mtcnn0 = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=80)

# Resnet model initiation
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(torch_device)




# INITIAL DATA LOADING ==============================================
def collate_fn(x):
    return x[0]

to_tensor = transforms.ToTensor()

name_list = [] # list of IDs
embedding_list = [] # list of embedings

dataset = datasets.ImageFolder(dataset_folder) # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names


loader = DataLoader(dataset, collate_fn=collate_fn)

for img, idx in loader:
    try:
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob > 0.75:
            # Ensure the face is a tensor
            if not isinstance(face, torch.Tensor):
                face = to_tensor(face)
            
            # Move tensor to device
            face = face.to(torch_device)
            
            # Check if resizing is necessary
            if face.size() != torch.Size([3, 160, 160]):
                face = transforms.functional.resize(face, (160, 160))
            
            # Forward pass through the ResNet model
            emb = resnet(face.unsqueeze(0))
            # Store the embedding and corresponding label
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])
    except Exception as e:
        print(f"Error processing image: {e}")
        continue

# save data.pt file
data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # saving data.pt file


# DATABASE UPDATE FUNCTIONS ==========================================
def update_db(embedding_list, name_list, photos_directory=dataset_folder):
    global mtcnn0, resnet, device, face_presence

    print("Update database function is being executed.")
    print(f"Current name list before update: {name_list}")
    #msg1 = f"Request to update db..."
    #send_msg(webhook, msg1)

    current_dataset = datasets.ImageFolder(photos_directory)
    current_indices = {v: k for k, v in current_dataset.class_to_idx.items()}
    print(f"Current indices from dataset: {current_indices}")

    new_entries = [name for idx, name in current_indices.items() if str(name) not in name_list]
    print(f"New entries detected based on name_list: {new_entries}")

    if not new_entries:
        print("No new folders detected.")
        return

    for name in new_entries:
        folder_path = os.path.join(photos_directory, name)
        print(f"Processing new folder: {name}")
        process_folder(folder_path, embedding_list, name_list, name)

    updated_data = [embedding_list, name_list]
    torch.save(updated_data, 'data.pt')
    face_presence = {folder_name: 0 for folder_name in next(os.walk(dataset_folder))[1]}
    print("Database updated with new faces.")
    print(f"Updated name list: {name_list}")
    #msg = f"Processed new database entry: {name}"
    #send_msg(webhook, msg)

def process_folder(folder_path, embedding_list, name_list, class_name):
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                face = mtcnn0(Image.fromarray(img))  # MTCNN called directly on the image
                if face is not None:
                    # Convert the tensor to a NumPy array and use CUDA for resizing
                    face = face.permute(1, 2, 0).cpu().numpy()  # Change CxHxW to HxWxC for cv2
                    gpu_face = cv2.cuda_GpuMat()
                    gpu_face.upload(face)
                    resized_gpu_face = cv2.cuda.resize(gpu_face, (160, 160))
                    face_tensor = to_tensor(resized_gpu_face.download()).to(device)

                    # Get embeddings and store them
                    emb = resnet(face_tensor.unsqueeze(0))
                    embedding_list.append(emb.detach())
                    name_list.append(str(class_name))  # Append name for every embedding
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

# FACE PRESENCE INITIATION ==========================================
def init_face_presence(directory=dataset_folder):
    # This function will create a dictionary with folder names as keys and 0 as values
    return {folder_name: 0 for folder_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder_name))}

face_presence = init_face_presence()

print(face_presence)

# OSC VALUES INITIATION ============================================
for name in face_presence.keys():
    osc_client.send_message(f"/{name}_X", 0)
    osc_client.send_message(f"/{name}_Y", 0)
    osc_client.send_message(f"/{name}_D", 0)
    osc_client.send_message(f"/{name}_Cam", 0)

# CV2 CAMERA INITIATION ============================================
cap = cv2.VideoCapture(camera_id)
threshold_distance = 0.9

cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

# LOG DISPLAY INITIATION ===========================================
#print("\n" * 5)  # line spacing
#display_log([], 0, "Ready", cv2_device, str(torch_device))


# LOADING DATA FROM DATA.PT FILE ===================================
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

# MAIN LOOP =======================================================
while cap.isOpened():

    # Resize windows if terminal dimensions change
    # Get the current window dimensions
    height, width = win.getmaxyx()
    
    # Recreate windows if resized
    if height < 20 or width < 1:
        break  # Optionally handle very small dimensions

    client_win = win.subwin(min(16, height - 3), width - 1, 3, 0)  # Adjusted to place below the title
    log_win = win.subwin(height - 19, width - 1, 19, 0)  # Adjusted for log window
    
    # Draw the title each iteration (to handle resizing)
    draw_title(win, "Face Recognition System")

    
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    current_detected_names = []  # Reset detected names
    detected_faces = []  # Reset detected faces
    faces, boxes = fast_mtcnn([frame_rgb])  # Detect faces
    
    unregistered_person_in_cam1 = False
    client_win.clear()


    if faces:
        gpu_faces = [cv2.cuda_GpuMat() for _ in faces] #Resize faces
        for gpu_face, face in zip(gpu_faces, faces):
            gpu_face.upload(face) #Upload faces to GPU
        try:
            # Resize and get face embeddings
            resized_gpu_faces = [cv2.cuda.resize(gpu_face, (160, 160)) for gpu_face in gpu_faces]
            face_tensors = [to_tensor(gpu_face.download()) for gpu_face in resized_gpu_faces]
            embeddings = resnet(torch.stack(face_tensors).to(torch_device))

            # Parse coordinates and embeddings
            for embedding, box in zip(embeddings, boxes[0]):
                x1, y1, x2, y2 = [int(b) for b in box]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                min_distance = threshold_distance
                min_name = None

                # Compare embeddings with database
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(embedding, emb_db).item()
                    if dist < min_distance:
                        min_distance = dist
                        min_name = name_list[idx]

                # Update values for any matching faces
                if min_name and min_distance <= threshold_distance:

                    osc_client.send_message(f"/{min_name}_Presence", 1)
                    osc_client.send_message(f"/{min_name}_X", center_x)
                    osc_client.send_message(f"/{min_name}_Y", center_y)
                    osc_client.send_message(f"/{min_name}_D", min_distance)
                    osc_client.send_message(f"/{min_name}_Cam", checkCamID(center_x, center_y))
                    detected_faces.append(min_name)
                    current_detected_names.append(min_name)

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Write name and distance
                    cv2.putText(frame, f"ID-{min_name} ({min_distance:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Known person check for check-in interface
                    if checkCamID(center_x, center_y) == 1:
                        unregistered_person_in_cam1 = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 250, 255), 2)
                    cv2.putText(frame, "??!!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 255), 2)

        # Error handling without interruption
        except Exception as e:
            status = f"An Error has occured: {e}"
            # Call display_interface without 'win' as an argument
            display_interface(client_win, log_win, faces, current_detected_names, status, cv2_device, str(torch_device))

            #client_win.addstr("Known faces in frame: {} \n Detected Faces: {}\n Status: {}\n OpenCV Device: {}\n Torch Device: {}".format(
            #current_detected_names, len(faces), status, cv2_device, str(torch_device)))            
            #display_log(current_detected_names, len(faces), f"An Error has occured: {e}", cv2_device, str(torch_device))
            continue

        status = "Detection completed"
        # Call display_interface without 'win' as an argument
        display_interface(client_win, log_win, faces, current_detected_names, status, cv2_device, str(torch_device))

        #client_win.addstr("Known faces in frame: {} \n Detected Faces: {}\n Status: {}\n OpenCV Device: {}\n Torch Device: {}".format(
        #current_detected_names, len(faces), status, cv2_device, str(torch_device)))

    else:
        status = "No faces detected"
        # Call display_interface without 'win' as an argument
        display_interface(client_win, log_win, faces, current_detected_names, status, cv2_device, str(torch_device))

        #client_win.addstr("Known faces in frame: {} \n Detected Faces: {}\n Status: {}\n OpenCV Device: {}\n Torch Device: {}".format(
        #current_detected_names, len(faces), status, cv2_device, str(torch_device)))         
        #display_log([], 0, "No faces detected", cv2_device, str(torch_device))

    client_win.refresh()

    #Reset presences for undetected faces
    for name in face_presence.keys():
        if name not in detected_faces:
            osc_client.send_message(f"/{name}_Presence", 0)
            face_presence[name] = 0
    if unregistered_person_in_cam1 == True:
        osc_client.send_message(f"/unregistered_person_in_cam1", 1)
    else:
        osc_client.send_message(f"/unregistered_person_in_cam1", 0)

    # Resize frame to match NDI res
    h, w, _ = frame.shape
    aspect_ratio = w / h
    target_width = min(ndi_width, int(ndi_height * aspect_ratio))
    if cv2_cuda_enabled:
        # If CUDA is available, use CUDA resizing
        cuda_frame = cv2.cuda_GpuMat()
        cuda_frame.upload(frame)
        cuda_resized_frame = cv2.cuda.resize(cuda_frame, (target_width, ndi_height))
        resized_frame = cuda_resized_frame.download()
    else:
        resized_frame = cv2.resize(frame, (target_width, ndi_height))
    
    # NDI OUTPUT =======================================================

    # Calculate padding
    pad_left = (ndi_width - target_width) // 2
    pad_right = ndi_width - target_width - pad_left

    # Pad resized frame to match exact res
    padded_frame = cv2.copyMakeBorder(resized_frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    frame_bgra = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2BGRA)

    # Set NDI video frame data
    video_frame.data = frame_bgra
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

    # Send frame over NDI
    ndi.send_send_video_v2(ndi_send, video_frame)
    if cv_show_render:
        cv2.imshow("Face Detection", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cleanup_ndi()
server.shutdown()
server.server_close()