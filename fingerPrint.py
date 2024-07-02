import cv2
import numpy as np
import os

def descriptores(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Não foi possível carregar a imagem!")
        return None, None

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

def Filetxt(file):
    return np.loadtxt(file, dtype=np.float32)

descriptor = "./Arquivos descritores das impressões digitais/file2.txt"
test_descriptors = Filetxt(descriptor)


match_points = []

bestImage = None
bestfile = None

for file in os.listdir("./Banco de Dados"):
      fingerprint_database_image_path = os.path.join("./Banco de Dados", file)
      fingerprint_database_image = cv2.imread(fingerprint_database_image_path, cv2.IMREAD_GRAYSCALE)
      cv2.imshow("Imagem banco", fingerprint_database_image)
      cv2.waitKey(5000)
      cv2.destroyAllWindows()

      if fingerprint_database_image is None:
          print(f"Não foi possível carregar a imagem!")
          continue
      
      keypoints1, descriptors = descriptores(fingerprint_database_image_path)

      if descriptors is None:
          continue

      
      flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
      matches = flann.knnMatch(test_descriptors, descriptors, k=2)

      
      Bestmatches = []
      for p, q in matches:
          if p.distance < 0.75 * q.distance:
              Bestmatches.append(p)

      
      if len(Bestmatches) / len(test_descriptors) > 0:
          match_points.extend(Bestmatches)
          bestImage = fingerprint_database_image
          bestfile = file
          break

if bestImage is not None:
    keypoints = len(test_descriptors)
    print(f"Keypoints: {keypoints}")
    print(f"Match Points: {len(match_points)}")
    print(f"Percentual de correspondentes: {len(match_points) / keypoints * 100:.2f}%")
    print(f"Fingerprint ID: {bestfile}")

    
    result = cv2.drawMatches(bestImage, keypoints1, bestImage, keypoints1, match_points, None)
    result = cv2.resize(result, None, fx=1.5, fy=1.5)
    cv2.imshow("Resultado", result)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

