import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

class SetCardAnalyzer(object):
    def __init__(self, debug, log, render_final):
        self.debug = debug
        self.should_log = log
        self.render_final = render_final
        with file('data/card.cnt') as f:
            self.CARD_CONTOUR = pickle.load(f)

    def find_set_cards(self, img):
        scene = self.prep_image(img)
        scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        
        final_scene = scene.copy()
        
        cards = self.extract_cards(scene)
        flat_cards = [self.flatten_card(c, scene) for c in cards]
        card_set = set()
        for (f,cnt) in zip(flat_cards, cards):
            if f != None:
                card = self.recognize_card(f)
                if card != None:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.putText(final_scene, str(card), (x, y+h + 5), cv2.FONT_HERSHEY_PLAIN, .5, (255,255,0), 1, cv2.LINE_AA)
                    card_set.add(card)
        if self.render_final:
            plt.imshow(final_scene),plt.show()
        return card_set
    
    def show(self, img):
        if self.debug:
            plt.imshow(img),plt.show()

    def log(self, s):
        if self.should_log:
            print s
        
    def extract_cards(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1,1), 1000)

        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,2)

        self.show(thresh)

        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[:20]

        CNT_THRESH = 0.2
        filtered = [cnt for cnt in contours if cv2.matchShapes(self.CARD_CONTOUR, cnt, 1, 0.0) < CNT_THRESH]
        copy = img.copy()
        cv2.drawContours(copy, filtered, -1, (0, 255, 0), 10)
        self.show(copy)
        return filtered

    def flatten_card(self, contour, image):
        width = 180
        height = 116 

        peri = cv2.arcLength(contour,True)
        # squeeze transforms the shape from (4,1,2) to (4,2)
        corners = np.squeeze(cv2.approxPolyDP(contour, 0.02*peri, True))
        ordered_corners = self.order_corners(corners).astype(np.float32)
        target = np.array([ [0,0],[0,width],[height,width],[height, 0] ],np.float32)
        if len(ordered_corners) != 4:
            return None
        transform = cv2.getPerspectiveTransform(ordered_corners,target)
        warp = cv2.warpPerspective(image,transform,(height,width))
        #show(warp)
        return warp

    def order_corners(self, corners):
        # Order the corners so that it is long edge -> short edge -> ...
        p0 = corners[0]
        p1 = corners[1]
        p2 = corners[2]
        if (self.dist(p0, p1) > self.dist(p1, p2)):
            return corners
        else:
            return np.append(corners[1:], [corners[0]], axis = 0)

    def dist(self, x,y):   
        return np.sqrt(np.sum((x-y)**2))

    def recognize_card(self, card):
        card = card[10:-10,10:-10]
        gray = cv2.cvtColor(card.copy(),cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 1000)
       

        flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # TODO: if you change the card side maybe also change this constant
        #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,201,2)

        self.show(thresh)

        im2, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        # The 0th contour is the card itself, could add a filter for that
        
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[0:6]
        ccard = card.copy()
        
        filtered = self.remove_interior_contours(contours)
        cv2.drawContours(ccard, filtered, -1, (0,255,0), 1)

        shapes = [(self.identify_shape(cnt), cnt) for cnt in filtered if self.identify_shape(cnt) != None]
       

        self.show(ccard)

        if len(shapes) > 0:
            ((head_shape, head_score), head_contour) = shapes[0]
            fill = self.identify_fill(thresh, head_contour)
            color = self.identify_color(card, head_contour)
            return (head_shape, color, fill, len(shapes))
        else:
            return None

    def rgbToYUV(self, r,g,b):
       y =  0.299 * r + 0.587 * g + 0.114 * b
       u = -0.147 * r - 0.289 * g + 0.436 * b
       v =  0.615 * r - 0.515 * g - 0.100 * b
       return (y,u,v)

    def remove_interior_contours(self, contours):
        bad = set() 
        for (i, cont) in enumerate(contours):
            for (j, internal_cont) in enumerate(contours[i:]):
                bb_ext = cv2.boundingRect(cont)
                bb_int = cv2.boundingRect(internal_cont)
                if self.bb_overlap(bb_ext, bb_int):
                    bad.add(i + j)

        return [c for (i, c) in enumerate(contours) if not i in bad]

    def bb_overlap(self, bb_ext, bb_int):
        x,y,w,h = bb_ext
        x1,y1,w1,h1 = bb_int
        return x < x1 and y < y1 and x + w > x1 + w1 and y + h > y1 + h1

    def identify_shape(self, cnt):
        THRESH = 0.15
        SHAPES = {
                "oval": self.load_contour("oval"),
                "diamond": self.load_contour("diamond"),
                "squiggle": self.load_contour("squiggle")
        }
        res = [(cv2.matchShapes(shape, cnt, 1, 0.0), name) for (name, shape) in SHAPES.iteritems()]
        score, shape = min(res)
        if score < THRESH:
            return (shape, score)
        else:
            return None

    def identify_fill(self, card_thresh, cnt):
        bb = cv2.boundingRect(cnt)
        average = np.average(self.center_section(card_thresh, bb))
        if average < 20:
            self.log("empty (fill: %s)" % average)
            return "empty"
        if 20 <= average < 240:
            self.log("striped (fill: %s)" % average)
            return "striped"
        else:
            self.log("solid (fill: %s)" % average)
            return "solid"

    def identify_color(self, color_card, cnt):
        # Threshold to find number of colored pictures
        # Sum number of colored pixels / threshold
        bb = cv2.boundingRect(cnt)
        figure = self.select(color_card, bb)
        yuv_fig = figure 
        gray = cv2.cvtColor(figure, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (1,1), 1000)
        flag, mask = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        
        pixels = cv2.bitwise_and(yuv_fig, yuv_fig, mask = inv_mask)
        self.show(pixels)
        gavg = np.sum(pixels[:,:,1]) / (np.sum(inv_mask) / 255)
        ravg = np.sum(pixels[:,:,0]) / (np.sum(inv_mask) / 255)
        bavg = np.sum(pixels[:,:,2]) / (np.sum(inv_mask) / 255)
        (y,u,v) = self.rgbToYUV(ravg, gavg, bavg)

        if u > 0 and v > 0:
            return "purple"
        elif u < 0 and v < 0:
            return "green"
        else:
            return "red"

    def select(self, img, bb):
        x,y,w,h = bb
        return img[y:y+h,x:x+w]
      

    def center_section(self, img, bb):
        x,y,w,h = bb
        cent_x = x + w/4
        cent_y = y + h/4
        cent_h = h/2
        cent_w = w/2
        return img[cent_y: cent_y + cent_h, cent_x:cent_x+cent_w]

        
    def load_contour(self, shape):
        with file('data/%s.cnt' % shape) as f:
            return pickle.load(f)

    def prep_image(self, image):
        Target_Width = 500
        (h, w, c) = image.shape

        scale = Target_Width / float(w)
        newH = int(h * scale)
        return cv2.resize(image, (Target_Width, newH), cv2.INTER_AREA)


def to_json(card_set):
    import json
    l = []
    for card in sorted(card_set):
        (shape, color, fill, number) = card
        l.append({"shape": shape, "color": color, "fill": fill, "number": number})
    return json.dumps(l, indent=2, sort_keys = True)

if __name__ == "__main__":
    import sys
    import argparse
    

    parser = argparse.ArgumentParser(description='Set card analyzer')
    parser.add_argument('-d', action="store_true", dest="debug", default=False, help="Debug mode")
    parser.add_argument('-l', action="store_true", dest="log", default=False, help="Debug mode")
    parser.add_argument('-f', action="store", dest="path")
    parser.add_argument('-r', action="store_true", dest="render_final")
    args = parser.parse_args()

    scene = cv2.imread(args.path)
    set_finder = SetCardAnalyzer(debug = args.debug, log = args.log, render_final = args.render_final)
    cards = set_finder.find_set_cards(scene)
    print to_json(cards)

