import cv2

def split(img, n):
    sectors = []
    width = img.shape[1]//n
    for i in range(n-1):
        sectors.append(img[:,i*width:(i+1)*width])
    sectors.append(img[:,(i+1)*width:])
    return sectors
def apply_single_transform(sector,transform):
    if transform == 'const':
        return sector
    if transform == 'canny':
        gray = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
def apply_transforms(frame, keep_orig, transforms):
    if keep_orig:
        sectors = split(frame, len(transforms)+1)
        transforms = ['const']+transforms
    else:
        sectors = split(frame, len(transforms))
    sectors_transformed = [apply_single_transform(sector, transform) for sector,transform in zip(sectors,transforms)]


if __name__ == "__main__":
    parameters = [
        {'duration':(1, 400), 'keep_orig':True, 'transforms':['bw', 'canny']},
    ]

    cap = cv2.VideoCapture('videos/test.mp4')
    frame_count = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            
            frame_count += 1
            print(frame_count)
            print(frame.shape)
            sectors = split(frame, 3)
            #sectors = [apply_single_transform(sector, transform) for sector,transform in zip(sectors,['const', 'canny'])]
            for i, sector in enumerate(sectors):
                print(i)
                cv2.imshow('sector {}'.format(i), sector)
            
            # for parameter in parameters:
            #     if frame_count>=parameter.duration[0] and frame_count<=parameter.duration[1]:
            #     frame = apply_transforms(frame, keep_orig = parameter.keep_orig, transforms = parameter.transforms)
                    


        
            cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
