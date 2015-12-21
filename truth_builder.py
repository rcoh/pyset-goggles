
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

#card = cv2.imread('training/oval.jpg',0) # trainImage

###############################################################################
# Utility code from 
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

def extract_cards(img):
    gray = cv2.cvtColor(scene ,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)

    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:20]

    CNT_THRESH = 0.05
    print [cv2.matchShapes(CARD_CONTOUR, cnt, 1, 0.0) for cnt in contours]
    filtered = [cnt for cnt in contours if cv2.matchShapes(CARD_CONTOUR, cnt, 1, 0.0) < CNT_THRESH]
    return filtered

def flatten_card(contour, image):
    width = 180
    height = 116 
    x,y,w,h = cv2.boundingRect(contour)
    if w > h:
        print "TODODO"
    peri = cv2.arcLength(contour,True)
    approx = rectify(cv2.approxPolyDP(contour, 0.02*peri, True))
    h = np.array([ [0,0],[height,0],[height,width],[0,width] ],np.float32)
    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(image,transform,(height,width))
    return warp

def recognize_card(card):
    gray = cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)

    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[1:2]
    cv2.drawContours(card, [contours[0]], -1, (0,255,0), 2)
    plt.imshow(card),plt.show()
    return contours[0]

    
    

if __name__ == "__main__":
    with file('data/card.cnt') as f:
        CARD_CONTOUR = pickle.load(f)

    scene = cv2.imread('training/striped-diamond.jpg')          # queryImage
    cards = extract_cards(scene)
    flat_cards = [flatten_card(c, scene) for c in cards]
    assert len(flat_cards) == 1

    for card in flat_cards:
        cnt = recognize_card(card)
        with file('data/diamond.cnt', 'w') as f:
            pickle.dump(cnt, f)

