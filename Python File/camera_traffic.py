from ctypes import *
from enum import IntFlag
from enum import IntEnum
import os, sys
import numpy as np
import cv2
import time

QDEEP = CDLL('./lib/qdeep/libQDEEP.so')
QDEEP_COLORSPACE_TYPE_BGR24 = c_ulong(1)
QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE = c_ulong(48) 
QDEEP_OBJECT_DETECT_FLAG_TRAJECTORY_TRACKING = c_ulong(1) 

class QDeepObjectDetectFlag(IntFlag):
    TRAJECTORY_TRACKING = 0x00000001
    SUB_CLASS           = 0x00000002  # ONLY USED: FACE LANDMARK (AGE & GENDER)
    FEATURE_VECTOR      = 0x00000004
    BEHAVIOR            = 0x00000008  # ONLY USED: FACE LANDMARK (EMOTION)
    EXTRA_ATTRIBUTE     = 0x00000010  # ONLY USED: FACE LANDMARK #5 (ROLL / YAW)
                                      # ONLY USED: FACE LANDMARK #68 (ROLL / YAW / PITCH)
    FULL                = TRAJECTORY_TRACKING | SUB_CLASS | FEATURE_VECTOR  # 0x00000007

    
class QDeepGPUType(IntFlag):
    DEFAULT            = 0x00000001
    NVIDIA             = 0x00000001
    INTEL_CPU          = 0x00000002
    INTEL_GPU          = 0x00000004
    INTEL_VPU_MOVIDIUS = 0x00000008  # MYRAID X / MYRAID 2 / NCS

class QDeepObjectDetectConfigModel(IntEnum):
    # |NVIDIA|INTEL|CLASS|SUB-CLA|KPS|BHS|IMG|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_AOI_GENERAL_DEFECT_DETECTION = 0  # |oooooo|ooooo|00001|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_AOI_PCB_DEFECT_DETECTION = 1      # |oooooo|ooooo|00001|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_AOI_GAUGE_READER_DETECTION = 2    # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_BOAT = 3                          # |oooooo|ooooo|00005|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CODESCAN_RECOGNITION_BARCODE = 4  # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CODESCAN_RECOGNITION_QRCODE = 5   # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_COMPARISON_AUDIO = 6              # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_COMPARISON_VIDEO = 7              # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_DEPTH_MAP_3D_EX = 8               # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_DRIVING_DISTRACTION = 9           # |oooooo|ooooo|00009|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_EDUCATION = 10                     # |oooooo|ooooo|00063|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_HEAD_BODY = 11                # |oooooo|ooooo|00004|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_5_KEYPOINTS = 12     # |oooooo|ooooo|00002|-------|005|008|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_68_KEYPOINTS = 13    # |oooooo|ooooo|00002|-------|076|008|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_68_KEYPOINTS_3D_EX = 14  # |oooooo|ooooo|00004|-------|076|008|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_68_KEYPOINTS_FACE_BEAUTY_EX = 15  # |oooooo|ooooo|00002|-------|076|008|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FACE_LANDMARK_68_KEYPOINTS_MODAL_ANALYTICS_EX = 16  # |oooooo|ooooo|00002|-------|076|003|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_FLAME = 17                         # |oooooo|ooooo|00002|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HAND_LANDMARK_21_KEYPOINTS = 18    # |oooooo|ooooo|00001|-------|021|009|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_BACKGROUND_BLURRING = 19     # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_BACKGROUND_REMOVAL = 20      # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_EPTZ_AUTO_FRAMING = 21       # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_EPTZ_FACE_LAYOUT = 22        # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_EPTZ_SPEAKER_TRACKING = 23   # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_HANDWRITE_EXTRACTION = 24    # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SAFETY_INSPECTION = 25       # |oooooo|ooooo|-----|1+1+1+1|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_17_KEYPOINTS = 26   # |oooooo|ooooo|00001|-------|017|007|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_HUMAN_SKELETON_136_KEYPOINTS = 27  # |oooooo|ooooo|00001|-------|136|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_LICENSE_PLATE_RECOGNITION_PARKING = 28  # |oooooo|ooooo|00001|-------|004|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_LICENSE_PLATE_RECOGNITION_LAW_ENFORCEMENT = 29  # |oooooo|ooooo|00001|-------|004|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_MISSING_OBJECT = 30                # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_OPTICAL_CHARACTER_RECOGNITION = 31   # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_PHOTO_CAPTION = 32                   # |oooooo|-----|-----|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_PHOTO_RETOUCHING = 33                # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_PHOTO_SUPER_RESOLUTION = 34          # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_RETAIL_PRODUCT_RECOGNITION = 35      # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_SECURITY_TAIWAN = 36                 # |oooooo|ooooo|00005|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_SEGMENTATION = 37                    # |oooooo|ooooo|-----|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_SPEECH_TRANSCRIBE_EN = 38            # |oooooo|-----|-----|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_SPEECH_TRANSCRIBE_JP = 39            # |oooooo|-----|-----|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_SPEECH_TRANSCRIBE_ZH = 40            # |oooooo|-----|-----|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_TRAFFIC = 41                         # |oooooo|ooooo|00008|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_TRAFFIC_TAIWAN_04 = 42               # |oooooo|ooooo|00004|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_TRAFFIC_TAIWAN_08 = 43               # |oooooo|ooooo|00008|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_UNATTENDED_OBJECT = 44               # |oooooo|ooooo|00001|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_VOICE_CONTROL_EN = 45                # |oooooo|-----|-----|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_XRAY_INSPECTION_SYSTEM = 46          # |oooooo|ooooo|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED = 47                      # |oooooo|ooooo|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_LITE = 48                 # |oooooo|ooooo|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_MEDICAL_GRADE = 49        # |oooooo|ooooo|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_CUSTOMIZED_MULTI_LABELS = 50         # |oooooo|ooooo|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_GENAI_EVERYTHING_DETECTION = 51      # |oooooo|-----|0000X|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_GENAI_EVERYTHING_SEGMENTATION = 52   # |oooooo|-----|0000X|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_NVIDIA_CLARA_AGX = 53                # |oooooo|     |ooooo|-------|---|---|ooo|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_YOYO_V5 = 54                         # |oooooo|-----|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_YOYO_V7 = 55                         # |oooooo|-----|ooooo|-------|---|---|---|
    QDEEP_OBJECT_DETECT_CONFIG_MODEL_RESERVED = 56

#對應位置qcap_linux.h
class QCAP_AV_FRAME_T(Structure):
	_fields_ = [("pData", c_ulong*8),
				("nPitch", c_int*8),
				("pPrivateData0", c_ulong),
				("nWidth", c_int),
				("nHeight", c_int),
				("nSamples", c_int),
				("nFormat", c_int)]
	
class QDEEP_OBJECT_DETECT_KEYPOINT(Structure):
	_fields_ = [("nX", c_ulong),
				("nY", c_ulong),
				("nZ", c_ulong),
				("fProbability", c_float)]

class QDEEP_OBJECT_DETECT_BOUNDING_BOX(Structure):
    _fields_ = [("nClassID", c_ulong),
				("nSubClassIDs", c_ulong*8),
				("nObjectID", c_ulong),
				("nX", c_ulong),
				("nY", c_ulong),
				("nWidth", c_ulong),
				("nHeight", c_ulong),
				("fProbability", c_float),
				("fSubProbabilities", c_float*8),
				("fExtraAttributes", c_float*8),
				("nTrajectoryXs", c_ulong*128),
				("nTrajectoryYs", c_ulong*128),
				("sKeypoints", QDEEP_OBJECT_DETECT_KEYPOINT*256),
				("fBehaviorVectors", c_float*32),
				("nBehaviorID", c_ulong),
				("fFeatureVectors", c_float*1024),
                ("nTimeStamp", c_ulonglong),
                ("nDuration", c_ulonglong),
				("pImageResultBuffer", c_void_p)]

class QuickReceiver():
    def __init__(self):

        ### AI ###
        strModelName = './model/taiwan_traffic_C4/QDEEP.OD.TAIWAN.TRAFFIC.C4.TINY.CFG'
        QDEEP.QDEEP_CREATE_OBJECT_DETECT(QDeepGPUType.NVIDIA, 0 , QDeepObjectDetectConfigModel.QDEEP_OBJECT_DETECT_CONFIG_MODEL_TRAFFIC_TAIWAN_08,
                                          strModelName.encode('utf-8'), byref(self.m_detector), 0, None)
        QDEEP.QDEEP_START_OBJECT_DETECT(self.m_detector)
        self.m_bStartDetector = 1
        self.m_nWidth = 1920
        self.m_nHeight = 1080

        fps_text = "FPS: 0"
        frame_count = 0
        start_time = time.time()
        # Initialize video capture
        video_type = input("Enter video file path or camera (e.g., 0 for video, 1 for webcam): ")

        try:
            video_type = int(video_type)
        except ValueError:
            print("Invalid input. Defaulting to webcam.")
            video_type = 1  # 預設使用攝影機

        if video_type == 0:
            video_path = "./model/taiwan_traffic_C4/car.mp4"
             # 確保檔案存在
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found at {video_path}")
                exit()
            self.cap = cv2.VideoCapture(video_path)
        else:
            try:
                video_input = int(input("Enter camera ID (e.g., 0 for webcam): "))
            except ValueError:
                print("Invalid camera ID. Defaulting to 0 (webcam).")
                video_input = 0  # 預設使用 0 號攝影機

            self.cap = cv2.VideoCapture(video_input)

        if not self.cap.isOpened():
            print("❌ Unable to open the video source. Check file path or camera connection.")
            exit()

        print("✅ Video source opened successfully!")

        # 設定解析度為 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 確認是否成功設定
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video Resolution: {actual_width}x{actual_height}")

        if actual_width != 1920 or actual_height != 1080:
            print("Warning: Could not set resolution to 1920x1080. Camera may not support it.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Fail to read")
                break

            ### AI analysis
            self.nObjectSize = c_ulong(1000)
            bufferlen = self.m_nWidth * self.m_nHeight * 3

            image_np = np.array(frame)
            pImageBuffer = cast(image_np.ctypes.data, c_void_p)

            QDEEP.QDEEP_SET_VIDEO_OBJECT_DETECT_UNCOMPRESSION_BUFFER(self.m_detector, QDEEP_COLORSPACE_TYPE_BGR24, c_ulong(self.m_nWidth), c_ulong(self.m_nHeight), pImageBuffer, c_ulong(bufferlen), self.m_pObjectList, byref(self.nObjectSize), 1)
            # print("self.nObjectSize is ", self.nObjectSize)

            for i in range(self.nObjectSize.value):
                    object_color = (0, 0, 255)
                    cv2.rectangle(frame, (self.m_pObjectList[i].nX, self.m_pObjectList[i].nY), (self.m_pObjectList[i].nX+self.m_pObjectList[i].nWidth, self.m_pObjectList[i].nY+self.m_pObjectList[i].nHeight), object_color, 1, cv2.LINE_AA)

                    #tracking path
                    pointcount = 0
                    for j in range(128):
                        if self.m_pObjectList[i].nTrajectoryXs[j] == 0 and self.m_pObjectList[i].nTrajectoryYs[j] == 0:
                            break
                        else:
                            self.m_nTrajectoryX[j] = self.m_pObjectList[i].nTrajectoryXs[j]
                            self.m_nTrajectoryY[j] = self.m_pObjectList[i].nTrajectoryYs[j]
                            pointcount = j

                    for i in range(pointcount):
                        cv2.line(frame, (self.m_nTrajectoryX[i], self.m_nTrajectoryY[i]), (self.m_nTrajectoryX[i + 1], self.m_nTrajectoryY[i + 1]), object_color, 2)

            frame_count += 1
            # 每 1 秒更新一次 FPS 文字
            if time.time() - start_time >= 1:
                fps_text = f"FPS: {frame_count}"
                frame_count = 0
                start_time = time.time()

            # 🔽 在畫面左上角顯示 FPS
            cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('DEMO', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()        
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        if self.m_bStartDetector == 1:
            self.m_bStartDetector = 0
            QDEEP.QDEEP_STOP_OBJECT_DETECT(self.m_detector)
            QDEEP.QDEEP_DESTROY_OBJECT_DETECT(self.m_detector)
            self.m_detector = c_void_p(0)            

    def __del__(self):
        self.close()

    m_detector = c_void_p(0)
    m_nWidth = c_ulong(0)
    m_nHeight = c_ulong(0)
    m_nTrajectoryX = (c_ulong * 128)()
    m_nTrajectoryY = (c_ulong * 128)()
    m_bStartDetector = 0
    nObjectSize = 1000
    PointerArrayType = QDEEP_OBJECT_DETECT_BOUNDING_BOX * nObjectSize
    m_pObjectList = PointerArrayType()

if  __name__ == '__main__':

    quick_receiver = QuickReceiver()

    quick_receiver.close()
