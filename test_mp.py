import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    if hasattr(mp, 'solutions'):
        print("mp.solutions is available")
        print(f"Holistic available: {hasattr(mp.solutions, 'holistic')}")
    else:
        print("mp.solutions is NOT available (Error)")
        print(f"Dir mp: {dir(mp)}")
except Exception as e:
    print(f"Error: {e}")
