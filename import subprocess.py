import subprocess
import sys

def install_packages():
    packages = ["opencv-python", "opencv-contrib-python", "cvzone", "mediapipe"]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}.")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}.")

if __name__ == "__main__":
    # Install required packages
    install_packages()

    # Test importing the packages
    try:
        import cv2
        import cvzone
        from cvzone.FaceMeshModule import FaceMeshDetector
        from cvzone.PlotModule import LivePlot
        import time

        print(f"Successfully imported OpenCV version: {cv2.__version__}")
        print("Successfully imported CVZone and its modules.")
    except ImportError as e:
        print(f"Error importing packages: {e}")

# run second section of code



import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time

# Initialize video capture and FaceMesh detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

# Define the Morse code dictionary
morse_code_dict = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', '.': '.-.-.-', ',': '--..--', '?': '..--..',
    "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
    '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.',
    '-': '-....-', '_': '..--.-', '"': '.-..-.', '@': '.--.-', ' ': '/'
}

# Initialize variables
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)
blink_start_time = None
dot_threshold = 350  # Short blink threshold in milliseconds (dot)
dash_threshold = 650  # Long blink threshold in milliseconds (dash)
letter_gap_threshold = 1700  # Time in ms to consider end of a letter
blinks = []
last_blink_time = None  # To track the time between blinks
eyes_closed = False  # Flag to check if the eyes are closed

def record_blink(start_time):
    global blink_duration, blinks
    blink_duration = (time.time() - start_time) * 1000  # Duration in milliseconds
    print(f"Blink Duration: {blink_duration} ms")  # Debugging information

    if blink_duration <= dot_threshold:  # Short blink (dot)
        blinks.append('.')
        print("Recorded a dot (.)")
    elif blink_duration >= dash_threshold:  # Long blink (dash)
        blinks.append('-')
        print("Recorded a dash (-)")
    
    blink_start_time = None

def morse_to_text(morse_code):
    inverted_dict = {value: key for key, value in morse_code_dict.items()}
    words = morse_code.strip().split(' / ')
    decoded_message = ''
    for word in words:
        characters = word.split(' ')
        for char in characters:
            decoded_message += inverted_dict.get(char, '?')
        decoded_message += ' '
    return decoded_message.strip()

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    current_time = time.time() * 1000  # Get current time in milliseconds

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and not eyes_closed:  # Eyes are now closed
            eyes_closed = True  # Set eyes_closed flag
            blink_start_time = time.time()  # Start measuring blink duration
            color = (0, 200, 0)
            blinkCounter += 1

        elif ratioAvg >= 35 and eyes_closed:  # Eyes have reopened
            eyes_closed = False  # Reset eyes_closed flag
            if blink_start_time is not None:
                record_blink(blink_start_time)  # Record blink duration after eyes reopen
                last_blink_time = current_time  # Record time of this blink
                blink_start_time = None
            color = (255, 0, 255)

        # If enough time has passed since the last blink, decode the letter
        if last_blink_time and (current_time - last_blink_time) > letter_gap_threshold:
            if blinks:
                morse_code = ''.join(blinks)
                decoded_message = morse_to_text(morse_code)
                print(f"Decoded Message: {decoded_message}")
                blinks = []  # Clear the list for the next letter
            last_blink_time = None  # Reset after decoding

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
