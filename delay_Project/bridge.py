import cv2
import numpy as np
import NDIlib as ndi
import time

def warp_flow(img, flow, steps=2.0):
    h, w = img.shape[:2]
    flow_u = flow[..., 0]
    flow_v = flow[..., 1]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    remap_u = (u + flow_u * steps).astype(np.float32)
    remap_v = (v + flow_v * steps).astype(np.float32)
    
    warped_img = cv2.remap(img, remap_u, remap_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped_img

def main():
    if not ndi.initialize():
        print("failed NDI，check NDI Tools。")
        return

    # 1. receiver 
    ndi_find = ndi.find_create_v2()
    print("looking for TouchDesigner image...")
    sources = []
    while not sources:
        ndi.find_wait_for_sources(ndi_find, 1000)
        sources = ndi.find_get_current_sources(ndi_find)
        time.sleep(0.5)
    
    ndi_recv_create = ndi.recv_create_v3()
    ndi.recv_connect(ndi_recv_create, sources[0])
    print(f"connected to TouchDesigner: {sources[0].ndi_name}")

    # 2. sender
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'Python_Future'
    ndi_send = ndi.send_create(send_settings)
    video_frame = ndi.VideoFrameV2()

    print("running .. Ctrl+C to stop.")

    prev_gray = None

    # 3. main loop 
    while True:
        t, v, _, _ = ndi.recv_capture_v2(ndi_recv_create, 1000)
        
        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data)
            ndi.recv_free_video_v2(ndi_recv_create, v)
            
            # ndi formats
            
            if len(frame.shape) == 3 and frame.shape[2] == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGRA_UYVY)
            
            # 4channels 
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            
            if prev_gray is None:
                prev_gray = curr_gray
                video_frame.data = frame
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                ndi.send_send_video_v2(ndi_send, video_frame)
                continue
            
            # compute optical
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )

            # predict warp
            processed_frame = warp_flow(frame, flow, steps=2.5)
            
            prev_gray = curr_gray

            # send back to TD
            video_frame.data = processed_frame
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA 
            ndi.send_send_video_v2(ndi_send, video_frame)

if __name__ == "__main__":
    main()