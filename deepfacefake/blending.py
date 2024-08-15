import cv2
import os,sys
import numpy as np

# insert at 1, 0 is the script path (or '' in REPL)
workingDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, workingDir)
from InsightFaceDectectRecognition import InsightFaceDectectRecognition
# # Initialize dlib's face detector and facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector=InsightFaceDectectRecognition(workingDir)
def get_landmarks(frame):
    
    (face,bbox,landmarkPts)=detector.DetectFace(frame)[0]
    
    return (np.array(landmarkPts),bbox)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray)
    # if len(rects) > 0:
    #     landmarks = predictor(gray, rects[0])
    #     return np.array([(p.x, p.y) for p in landmarks.parts()])
    # return None


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Ensure ROI dimensions are within image bounds
    r1 = [max(0, r1[0]), max(0, r1[1]), min(r1[2], img1.shape[1] - r1[0]), min(r1[3], img1.shape[0] - r1[1])]
    r2 = [max(0, r2[0]), max(0, r2[1]), min(r2[2], img2.shape[1] - r2[0]), min(r2[3], img2.shape[0] - r2[1])]

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    
    img2_rect = img2_rect * mask
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []

    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        ind = []
        for j in range(3):
            for k in range(len(points)):
                if abs(pt1[0] - points[k][0]) < 1 and abs(pt1[1] - points[k][1]) < 1:
                    ind.append(k)
                elif abs(pt2[0] - points[k][0]) < 1 and abs(pt2[1] - points[k][1]) < 1:
                    ind.append(k)
                elif abs(pt3[0] - points[k][0]) < 1 and abs(pt3[1] - points[k][1]) < 1:
                    ind.append(k)

        if len(ind) == 3:
            delaunay_triangles.append((ind[0], ind[1], ind[2]))

    return delaunay_triangles

def blendingImage(faceimage1,faceimage2):

# faceimage2 = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/areafake.png')
# faceimage1 = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/keeped.png')
# Get landmarks
    landmarks1,bbox1 = get_landmarks(faceimage1)
    landmarks2,bbox2 = get_landmarks(faceimage2)

    if landmarks1 is not None and landmarks2 is not None:
        rect = cv2.boundingRect(np.array(landmarks1))
        delaunay_triangles = calculate_delaunay_triangles(rect, landmarks1)

        for triangle in delaunay_triangles:
            t1 = [landmarks1[i] for i in triangle]
            t2 = [landmarks2[i] for i in triangle]

            warp_triangle(faceimage2, faceimage1, t2, t1)

        # Blending the face using seamless clone
        mask = np.zeros_like(faceimage1)
        convexhull = cv2.convexHull(np.array(landmarks1, dtype=np.float32))
        cv2.fillConvexPoly(mask, np.int32(convexhull), (255, 255, 255))

        r = cv2.boundingRect(convexhull)
        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        result = cv2.seamlessClone(faceimage2, faceimage1, mask, center, cv2.NORMAL_CLONE)
        
        x,y,w,h=bbox1
        oareaface=[(x,y),(x+w,y)]
        oareaface.extend(landmarks1[:32])
        
        result=cv2.cvtColor(result,cv2.COLOR_BGR2BGRA)
        
        result=makeTransparent(result,oareaface,10)
        result= detector.keepInsideArea(result,oareaface)
                

        # cv2.imshow("Result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return (result,landmarks1,landmarks2,bbox1,bbox2)
    else:
        # print("Could not find landmarks in one of the images.")
        return None
def makeTransparent(image,points,radius = 20, opacity=0.5):
    
    points = detector.sortPointsClockwise(points)
    # # Create a mask with the same dimensions as the image
    # mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

    # # Apply the transparency effect to each point
    # for (cx, cy) in points:
    #     cv2.circle(mask, (cx, cy), radius, (0), thickness=-1)

    # # Convert the mask to a 4-channel image with transparency
    # mask = cv2.merge([mask, mask, mask,mask/2])

    # # Blend the mask with the image
    # result = cv2.bitwise_and(image, mask)
        
        
    # # Create a mask for the alpha channel
    # alpha_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

    # # Apply transparency effect to each point
    # for (cx, cy) in points:
    #     cv2.circle(alpha_mask, (cx, cy), radius, (0), thickness=-1)

    # # Create a copy of the alpha channel
    # alpha_channel = image[:, :, 3].copy()

    # # Reduce the alpha channel value in the masked areas to 50% opacity
    # alpha_channel[alpha_mask == 0] = (alpha_channel[alpha_mask == 0] * opacity).astype(np.uint8)

    # # Update the image's alpha channel
    # image[:, :, 3] = alpha_channel
    
        
    # Create a mask for the path
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the path on the mask
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], color=255, thickness=radius)

    # Create a copy of the alpha channel
    alpha_channel = image[:, :, 3].copy()

    # Apply transparency effect to the path area
    alpha_channel[mask == 255] = (alpha_channel[mask == 255] * opacity).astype(np.uint8)

    # Update the image's alpha channel
    image[:, :, 3] = alpha_channel
    
    return image

def blendingPoints(background,points,blend_radius = 10):
    height, width = background.shape[:2]
    # Create a mask for blending
    blend_mask = np.zeros((height, width), dtype=np.uint8)

    # Create a mask with circles at the points
    for point in points:
        cv2.circle(blend_mask, point, blend_radius, 255, thickness=cv2.FILLED)

    # Create a blending effect (e.g., radial gradient)
    blend_effect = np.zeros((height, width, 3), dtype=np.uint8)
    
    alpha = 0.1  # Weight of the first image (source)
    beta = 1 - alpha  # Weight of the second image (target)
    
    # blended_image = cv2.addWeighted(imgsrc, alpha, imgoverlay, beta, 0)
    for point in points:
        for i in range(blend_radius):
            alpha1 = i / blend_radius
            color = (0, 255, 0)  # Green color for blending
            circle_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(circle_mask, point, blend_radius - i, 255, thickness=cv2.FILLED)
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            color_mask[:, :] = color
            # blend_effect = cv2.addWeighted(blend_effect, 1, color_mask, alpha, 0)
            
            blend_effect = cv2.addWeighted(blend_effect, alpha1, color_mask, 1-alpha1, 0)
            blend_effect = cv2.bitwise_and(blend_effect, cv2.merge([circle_mask, circle_mask, circle_mask]))

    # Apply the blending effect to the background
    blended_image = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(blend_mask))
    
    blended_image = cv2.addWeighted(blended_image, alpha, blend_effect, beta, 0)
    
    return blended_image


# import cv2
# import numpy as np




# # Read the images
# image1 = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/ducmnd.jpg')
# image2 = cv2.imread('/work/llm/Ner_Llm_Gpt/deepfacefake/hoandung.jpg')


# # Sample points1 and points2
# points1 = np.array([[400, 400] , [800, 400], [800, 800], [400, 800]], dtype=np.float32)
# points2 = np.array([[500, 500], [900, 500], [900, 900], [500, 900]], dtype=np.float32)
# l1, b1 = get_landmarks(image1)
# l2, b2 = get_landmarks(image2)

# # print(l1[:32])

def transitionBbox(b1,b2, image1,image2):

    points1 = np.array([[b1[0], b1[1]] , [b1[0]+b1[2], b1[1]], [b1[0]+b1[2], b1[1]+b1[3]], [b1[0], b1[1]+b1[3]]], dtype=np.float32)
    points2 = np.array([[b2[0], b2[1]] , [b2[0]+b2[2], b2[1]], [b2[0]+b2[2], b2[1]+b2[3]], [b2[0], b2[1]+b2[3]]], dtype=np.float32)

    # Get the perspective transform matrix to transform points2 to points1
    M = cv2.getPerspectiveTransform(points2, points1)

    # Warp the perspective of image2 to align it with points1
    warped_image = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

    # Create a mask of the warped image
    mask = np.zeros_like(image1, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points1.astype(np.int32), (255, 255, 255))

    # Extract the region of interest in image1 and combine it with the warped image
    image1_bg = cv2.bitwise_and(image1, cv2.bitwise_not(mask))
    result = cv2.add(image1_bg, cv2.bitwise_and(warped_image, mask))
    
    return result

    # # Display the result
    # cv2.imshow('Overlay Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Save the result if needed
    # cv2.imwrite('overlay_result.png', result)
