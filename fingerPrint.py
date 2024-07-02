import cv2
import numpy as np
import os


test_original = cv2.imread("./Banco de Dados/104_6.tif")
cv2.imshow("Teste",test_original)
cv2.waitKey(10000)
cv2.destroyAllWindows()

match_points = []
for file in [file for file in os.listdir("./Banco de Dados/")]:
    print(file)
    fingerprint_database_image = cv2.imread(f"./Banco de Dados/{file}")
    
    sift = cv2.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
    dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    
   

    for p, q in matches:
      if p.distance < 0.75*q.distance:
          match_points.append(p)
for file in [file for file in os.listdir("./Arquivos descritores das impressões digitais/")]: 
    descritores = np.loadtxt("./Arquivos descritores das impressões digitais/"+file+"/file"+file, dtype=float)



keypoints = 0
if len(keypoints_1) <= len(keypoints_2):
  keypoints = len(keypoints_1)      
else:
  keypoints = len(keypoints_2)
print(keypoints)  
print(len(match_points))
if (len(match_points) / keypoints)>0.3:
  print("% match: ", len(match_points) / keypoints * 100)
  print("Figerprint ID: " + str(file)) 
  result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None) 
  result = cv2.resize(result, None, fx=2.5, fy=2.5)
  cv2.imshow(result)
  cv2.waitKey(10000)