import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

###############################################################################
# Utility code from 
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

DEBUG = True

def show(img):
    if DEBUG:
        plt.imshow(img),plt.show()
    
def extract_cards(img):
    gray = cv2.cvtColor(scene ,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)

    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:20]

    CNT_THRESH = 0.07
    print [cv2.matchShapes(CARD_CONTOUR, cnt, 1, 0.0) for cnt in contours]
    filtered = [cnt for cnt in contours if cv2.matchShapes(CARD_CONTOUR, cnt, 1, 0.0) < CNT_THRESH]
    cv2.drawContours(img, filtered, -1, (0, 255, 0), 2)
    show(img)
    return filtered

def flatten_card(contour, image):
    width = 180
    height = 116 

    peri = cv2.arcLength(contour,True)
    # squeeze transforms the shape from (4,1,2) to (4,2)
    corners = np.squeeze(cv2.approxPolyDP(contour, 0.02*peri, True))
    ordered_corners = order_corners(corners).astype(np.float32)
    target = np.array([ [0,0],[0,width],[height,width],[height, 0] ],np.float32)
    
    transform = cv2.getPerspectiveTransform(ordered_corners,target)
    warp = cv2.warpPerspective(image,transform,(height,width))
    #show(warp)
    return warp

def order_corners(corners):
    # Order the corners so that it is long edge -> short edge -> ...
    p0 = corners[0]
    p1 = corners[1]
    p2 = corners[2]
    if (dist(p0, p1) > dist(p1, p2)):
        return corners
    else:
        return np.append(corners[1:], [corners[0]], axis = 0)

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def recognize_card(card):
    
    SHAPES = {
            "oval": load_contour("oval"),
            "diamond": load_contour("diamond"),
            "squiggle": load_contour("squiggle")
    }

    gray = cv2.cvtColor(card.copy(),cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)

    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # TODO: if you change the card side maybe also change this constant
    #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,201,2)

    show(thresh)

    im2, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # The 0th contour is the card itself, could add a filter for that
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[1:4]
    filtered = remove_interior_contours(contours)

    shapes = [(identify_shape(cnt), cnt) for cnt in filtered if identify_shape(cnt) != None]
    ccard = card.copy()
    cv2.drawContours(ccard, filtered, -1, (0,255,0), 2)
    show(ccard)
    if len(shapes) > 0:
        ((head_shape, head_score), head_contour) = shapes[0]
        fill = identify_fill(thresh, head_contour)
        color = identify_color(card, head_contour)
        return (head_shape, color, fill, len(shapes))
    else:
        return None

def remove_interior_contours(contours):
    bad = set() 
    for (i, cont) in enumerate(contours):
        for (j, internal_cont) in enumerate(contours[i:]):
            bb_ext = cv2.boundingRect(cont)
            bb_int = cv2.boundingRect(internal_cont)
            if bb_overlap(bb_ext, bb_int):
                print bb_ext, bb_int, i, i+j
                bad.add(i + j)

    return [c for (i, c) in enumerate(contours) if not i in bad]

def bb_overlap(bb_ext, bb_int):
    x,y,w,h = bb_ext
    x1,y1,w1,h1 = bb_int
    return x < x1 and y < y1 and x + w > x1 + w1 and y + h > y1 + h1

def identify_shape(cnt):
    THRESH = 0.07
    SHAPES = {
            "oval": load_contour("oval"),
            "diamond": load_contour("diamond"),
            "squiggle": load_contour("squiggle")
    }
    res = [(cv2.matchShapes(shape, cnt, 1, 0.0), name) for (name, shape) in SHAPES.iteritems()]
    score, shape = min(res)
    if score < THRESH:
        return (shape, score)
    else:
        print shape,score, "MISS"
        return None

def identify_fill(card_thresh, cnt):
    bb = cv2.boundingRect(cnt)
    average = np.average(center_section(card_thresh, bb))
    print "average: ", average
    if average < 10:
        return "solid"
    if 10 <= average < 200:
        return "striped"
    else:
        return "empty"

def identify_color(color_card, cnt):
    # Threshold to find number of colored pictures
    # Sum number of colored pixels / threshold
    bb = cv2.boundingRect(cnt)
    figure = select(color_card, bb)
    yuv_fig = figure 
    gray = cv2.cvtColor(figure, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)
    flag, mask = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    
    pixels = cv2.bitwise_and(yuv_fig, yuv_fig, mask = inv_mask)
    show(pixels)
    ravg = np.sum(pixels[:,:,0]) / (np.sum(inv_mask) / 255)
    gavg = np.sum(pixels[:,:,1]) / (np.sum(inv_mask) / 255)
    bavg = np.sum(pixels[:,:,2]) / (np.sum(inv_mask) / 255)
    print ravg,gavg,bavg
    if ravg > 100 and bavg > 100 and gavg < 110:
        return "purple"
    if gavg > 100:
        return "green"
    else:
        return "red"

    

def select(img, bb):
    x,y,w,h = bb
    return img[y:y+h,x:x+w]
  

def center_section(img, bb):
    x,y,w,h = bb
    cent_x = x + w/4
    cent_y = y + h/4
    cent_h = h/2
    cent_w = w/2
    return img[cent_y: cent_y + cent_h, cent_x:cent_x+cent_w]

    
def load_contour(shape):
    with file('data/%s.cnt' % shape) as f:
        return pickle.load(f)

if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    with file('data/card.cnt') as f:
        CARD_CONTOUR = pickle.load(f)

    scene = cv2.imread('training/%s.jpg' % name)          # queryImage
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    cards = extract_cards(scene)
    flat_cards = [flatten_card(c, scene) for c in cards]

    card_set = set()
    for (f,cnt) in zip(flat_cards, cards):
        card = recognize_card(f)
        x,y,w,h= cv2.boundingRect(cnt)
        print x,y,w,h
        cv2.putText(scene, str(card), (x, y+h + 20), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,0), 4, cv2.LINE_AA)
        print card
        card_set.add(card)

    plt.imshow(scene),plt.show()
    

    truth_path = "ground_truth/%s" % name
    if not os.path.isfile(truth_path):
        if raw_input(cards) == "y":
            with file(truth_path, 'w') as truth_file:
                pickle.dump(cards, truth_file)

    with file(truth_path) as truth_file:
        truth = pickle.load(truth_file) 
        assert card_set == truth
        print "Success!"
