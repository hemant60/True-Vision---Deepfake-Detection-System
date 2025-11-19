from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from app.verify import authentication
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
import os
import numpy as np
from django.shortcuts import render
from ultralytics import YOLO
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import subprocess
import imageio

# Load YOLOv8 model
model = YOLO('Dataset/train2/weights/best.pt')  # Replace with your YOLOv8 trained model path

##############################################################################
#                               Main Section                                 #
##############################################################################




def index(request):
    context = {
        'page' : 'home'
    }
    # return HttpResponse("This is Home page")    
    return render(request, "index.html", context)

def log_in(request):
    context = {
        'page' : 'log_in'
    }
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html", context)

def register(request):
    context = {
        'page' : 'register'
    }
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html", context)


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname': request.user.first_name, 
        
        }
    if request.method == 'POST' and 'image_upload' in request.POST:
        image = request.FILES['image']
        image_path = default_storage.save('uploaded_images/' + image.name, ContentFile(image.read()))

        # Read the image
        img = cv2.imread(default_storage.path(image_path))

        # Perform detection
        results = model.predict(source=img, save=False)

        # Annotate the image with bounding boxes
        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)  # Convert to integer
                conf = result.boxes.conf[i]  # Confidence score
                cls = int(result.boxes.cls[i])  # Class index
                label = f"{model.names[cls]} {conf:.2f}"

                box_color = (0, 0, 255) if "fake" in model.names[cls].lower() else (0, 255, 0)
                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image
        processed_image_path = 'media/detected_images/' + image.name
        cv2.imwrite(processed_image_path, img)

        # Send image path to frontend
        context['image'] = processed_image_path
        context['label'] = label
    
    if request.method == 'POST' and 'video_upload' in request.POST:
        video = request.FILES['uploaded_video']

        # Save uploaded video
        video_path = default_storage.save('uploaded_videos/' + video.name, ContentFile(video.read()))
        video_full_path = default_storage.path(video_path)

        # Define output path
        detected_video_path = os.path.join(settings.MEDIA_ROOT, 'detected_videos', video.name.replace('.mp4', '_detected.mp4'))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(detected_video_path), exist_ok=True)

        try:
            # Read input video
            reader = imageio.get_reader(video_full_path)
            fps = reader.get_meta_data()['fps']

            # Define output video writer
            writer = imageio.get_writer(detected_video_path, fps=fps)

            for frame in reader:
                # Convert image to OpenCV format (BGR)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Perform YOLOv8 detection
                results = model.predict(frame_bgr)

                # Annotate frame with bounding boxes
                for result in results:
                    for i, box in enumerate(result.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        conf = result.boxes.conf[i]
                        cls = int(result.boxes.cls[i])
                        label = f"{model.names[cls]} {conf:.2f}"
                        print(label)

                        # Determine bounding box color
                        box_color = (0, 0, 255) if "fake" in model.names[cls].lower() else (0, 255, 0)

                        # Draw bounding box and label
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Convert OpenCV BGR image back to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Write frame to output video
                writer.append_data(frame_rgb)

            writer.close()

            # Provide the detected video path to the frontend
            context['video'] = settings.MEDIA_URL + 'detected_videos/' + os.path.basename(detected_video_path)
            messages.success(request, "Video processed successfully with YOLOv8!")

        except Exception as e:
            messages.error(request, f"Error processing video: {str(e)}")
            
    if request.method == 'POST' and 'live_cam' in request.POST:
        # Open webcam (0 = default webcam)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 prediction
            results = model.predict(frame)

            for result in results:
                for i, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    conf = result.boxes.conf[i]
                    cls = int(result.boxes.cls[i])
                    label = f"{model.names[cls]} {conf:.2f}"

                    # Determine color
                    box_color = (0, 0, 255) if "fake" in model.names[cls].lower() else (0, 255, 0)

                    # Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Show annotated frame
            cv2.imshow('Live Detection', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    return render(request, "dashboard.html", context)