# This script is responsible for generating artificial brazilian plates
# and performing augmentation (consider real data styles).
import csv

from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
import time
import matplotlib.pyplot as plt
import random
import os
import sys
import collections
import imgaug as ia
import numpy as np

class PlateGenerator:
    def __init__(self, showPlates=True, showStatistics=False, augmentation=True, bgInsertion=False, contourOnly=False, isMercosul=True, isMotorcycle=False, isRed=False):
        self.dataFolder       = 'data/mercosul' if isMercosul else ('data/red' if isRed else 'data')
        self.letters          = ["A", "B", "C", "D", "E", "F", "G",
                                 "H", "I", "J", "K", "L", "M", "N",
                                 "O", "P", "Q", "R", "S", "T", "U",
                                 "V", "Y", "W", "X", "Z"]

        self.statistics       = collections.OrderedDict([("A",0), ("B",0), ("C",0), ("D",0), ("E",0),
                                                         ("F",0), ("G",0), ("H",0), ("I",0), ("J",0),
                                                         ("K",0), ("L",0), ("M",0), ("N",0), ("O",0),
                                                         ("P",0), ("Q",0), ("R",0), ("S",0), ("T",0),
                                                         ("U",0), ("V",0), ("Y",0), ("W",0), ("X",0),
                                                         ("Z",0), ("0",0), ("1",0), ("2",0), ("3",0),
                                                         ("4",0), ("5",0), ("6",0), ("7",0), ("8",0),
                                                         ("9",0), ("-",0), ("plate",0)])
        self.bboxes            = []
        self.bgFiles           = []
        self.numbers           = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.nLetters          = 3
        self.nNumbers          = 4
        self.isMercosul        = isMercosul
        self.isMotorcycle = isMotorcycle
        if self.isMotorcycle:
            if self.isMercosul:
                self.plateSize         = (128, 149)
                self.initialWidth = 110
                self.initialHeight = 100
                self.widthRef = 70
                self.heightRef = 85
                self.charPadding = 10
            else:
                self.plateSize         = (128, 149)
                self.initialWidth = 50
                self.initialHeight = 100
                self.widthRef = 70
                self.heightRef = 85
                self.charPadding = 5
        else:
            self.plateSize         = (212, 646)
            if self.isMercosul:
                self.initialWidth = 70
                self.initialHeight = 75
                self.widthRef = 70
                self.heightRef = 85
                self.charPadding = 4
            else:
                self.plateSize         = (40, 108)
                self.plateSize         = (162, 500)
                self.initialWidth = 30
                self.initialHeight = 55
                self.widthRef = 30
                self.heightRef = 55
                self.charPadding = 0

        self.resizeBackground  = (800, 600)
        self.resizePlateFactor = 'random'
        self.centerPlate       = False
        self.contourOnly       = contourOnly
        self.visualizePlates   = showPlates
        self.bgInsertion       = bgInsertion
        self.augmentation      = augmentation
        self.bgFolder          = '../images/test-plates/'
        self.showStatistics    = showStatistics
        self.plateSample       = os.path.join(self.dataFolder, 'plate-motorcycle.jpg' if self.isMotorcycle else 'plateSample01.jpg')
        self.plateIm           = Image.open(self.plateSample)
        self.resetReferences()

        # get possible background images
        for root, dirs, files in os.walk(self.bgFolder):
            for name in files:
                self.bgFiles.append(os.path.join(root, name))


    def resetReferences(self):
        self.widthRef  = self.initialWidth
        self.heightRef = self.initialHeight
        self.bboxes    = []

    def generateLetters(self, image, quantity=None):
        # Adding letters
        if quantity is None:
            quantity = self.nLetters
        for _ in range(0, quantity):
            randomChar = random.choice(self.letters)
            file = randomChar

            if not self.isMercosul:
                if randomChar == "I":
                    file = "I1"

                if randomChar == "O":
                    file = "O0"

            char = Image.open(os.path.join(self.dataFolder, "%s.png" % str(file)))
            padding = self.charPadding
            if self.isMercosul:
                char = char.resize((70,110))
            if self.isMotorcycle and not self.isMercosul:
                char = char.resize((90,130))
                padding = int(padding * 6.5)

            charW, charH = char.size

            annotations = self.generateBox(charW, charH, str(randomChar))
            if not self.contourOnly:
                # Append box according to widthRef + charW
                self.bboxes.append(annotations)

            image.paste(char, (self.widthRef, self.heightRef), char)
            self.widthRef += charW + padding

            # Increment statistics
            self.statistics[randomChar] +=1
        return image

    def generateBox(self, charW, charH, tag):
        xMin = self.widthRef
        yMin = self.heightRef
        xMax = self.widthRef + charW + self.charPadding
        yMax = self.heightRef + charH
        return xMin,yMin, xMax, yMax, tag

    def generateNumbers(self, image, quantity=None):
        # Adding numbers
        if quantity is None:
            quantity = self.nNumbers
        for _ in range(0, quantity):
            randomNum = random.choice(self.numbers)
            file = randomNum
            if not self.isMercosul:
                if randomNum == "1":
                    file = "I1"

                if randomNum == "0":
                    file = "O0"

            number = Image.open(os.path.join(self.dataFolder, "%s.png" % str(file)))
            if self.isMercosul:
                number = number.resize((70,110))

            if self.isMotorcycle and not self.isMercosul:
                number = number.resize((80, 120))
            numberW, numberH = number.size

            annotations = self.generateBox(numberW, numberH, randomNum)
            if not self.contourOnly:
                # Append box according to widthRef + numberW
                self.bboxes.append(annotations)

            image.paste(number, (self.widthRef, self.heightRef), number)
            self.widthRef += numberW + self.charPadding

            # Increment statistics
            self.statistics[str(randomNum)] +=1
        return image

    def generateDash(self, image, includeDash):
        # Adding dash
        dash = Image.open(os.path.join(self.dataFolder, "%s.png" % str("-")))
        dashW, dashH = dash.size

        if includeDash and not self.contourOnly:
            # Append box according to widthRef + numberW
            self.bboxes.append(self.generateBox(dashW, dashH, "-"))

        image.paste(dash, (self.widthRef, self.heightRef), dash)
        self.widthRef += dashW + self.charPadding

        # Increment statistics
        self.statistics["-"] += 1
        return image

    def visualizePlate(self, image, bboxes):
        draw = ImageDraw.Draw(image)
        for box in bboxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], None, (0,255,0))
            draw.rectangle([(box[0]-1, box[1]-1), (box[2]+1, box[3]+1)], None, (0,255,0))
        plt.imshow(image)
        plt.show()


    def generatePlates(self, numOfPlates, trainSet=True, includeDash=False, resize=True):
        print("------------------------------------------------------------------")
        print("Generating Artificial Data...")
        startTime = time.time()
        plates    = []
        for idx in range(0, numOfPlates):
            plateSample   = self.generatePlateBackground()
            if self.isMercosul:
                finalImg             = self.generateLetters(plateSample, 3)
                if self.isMotorcycle:
                    self.nextLine()
                finalImg             = self.generateNumbers(finalImg, 1)
                if random.randint(0,1) == 1:
                    finalImg             = self.generateLetters(finalImg, 1)
                else:
                    finalImg             = self.generateNumbers(finalImg, 1)
                finalImg             = self.generateNumbers(finalImg, 2)
            else:
                finalImg             = self.generateLetters(plateSample)
                if self.isMotorcycle:
                    self.nextLine()
                # finalImg             = self.generateDash(finalImg, includeDash)
                finalImg             = self.generateNumbers(finalImg)

            # Perform data augmentation
            if self.augmentation:
                img, boxes = self.augmentImg({"plateImg": finalImg, "plateBoxes": self.bboxes}, resize=resize)
            else:
                img = finalImg
                boxes = self.bboxes

            if self.bgInsertion:
                augBoxes = []

                backgroundFile = random.choice(self.bgFiles)
                bgImg = Image.open(backgroundFile)
                bgImg = bgImg.resize(self.resizeBackground, Image.ANTIALIAS)

                bgW, bgH = bgImg.size
                plateW, plateH = img.size
                offset = (int((bgW - plateW) * random.uniform(0.1, 1.0)), int((bgH - plateH) * random.uniform(0.1, 1.0)))

                if self.centerPlate:
                    offset = ((bgW - plateW) // 2, (bgH - plateH) // 2)

                for box in boxes:
                    cls  = box[4]

                    xMin = box[0] + offset[0]
                    yMin = box[1] + offset[1]
                    xMax = box[2] + offset[0]
                    yMax = box[3] + offset[1]
                    augBoxes.append((xMin, yMin, xMax, yMax, cls))
                boxes = augBoxes
                bgImg.paste(img, offset)
                img = bgImg
            if self.visualizePlates:
                self.visualizePlate(img, boxes)

            plates.append({"plateIdx": idx, "plateImg": img, "plateBoxes": boxes})
            # Reset references (width, height and boxes)
            self.resetReferences()

        # Show histogram
        if self.showStatistics:
            self.visualizeStatistics()

        elapsed = round((time.time() - startTime),3)

        print("Plates generated succesfully in %s seconds" % str(elapsed))
        return plates

    def generatePlateBackground(self):
        plateSample = self.plateIm.copy()
        plateW, plateH = plateSample.size
        xMin = 0
        yMin = 0
        xMax = plateW
        yMax = plateH
        self.bboxes.append((xMin, yMin, xMax, yMax, "plate"))
        return plateSample

    def visualizeStatistics(self):
        plt.figure()
        plt.title("Characters Histogram")
        plt.bar(self.statistics.keys(), self.statistics.values(), 1, color='g')
        plt.show()

    def augmentImg(self, plate, resize=False):
        # for plate in plates:
        plateImg    = np.asarray(plate['plateImg'])
        plateBoxes  = plate['plateBoxes']
        bbs         = []

        for box in plateBoxes:
            bbs.append(ia.BoundingBox(box[0], box[1], box[2], box[3]))
        bboxAug = ia.BoundingBoxesOnImage(bbs, shape=plateImg.shape)

        if resize:
            if self.resizePlateFactor == 'random':
                resizeFactorW = round(random.uniform(0.3, 0.9), 1)

                ratio = self.plateSize[0] / self.plateSize[1]
                newWidth  = resizeFactorW*self.plateSize[0]
                newHeight = newWidth/ratio
                plateSize = (int(newWidth), int(newHeight))
                # plateSize = (int(self.plateSize[0]), int(self.plateSize[1]))

            # Rescale image and bounding boxes
            plateImg = ia.imresize_single_image(plateImg, plateSize)
            # print(plateSize)
            bboxAug = bboxAug.on(plateImg)

        seq    = iaa.Sequential([
            iaa.Sometimes(0.6,
                          iaa.OneOf([iaa.GaussianBlur((0, 0.8)) # blur images with a sigma between 0 and 1.0
                                     # iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 5
                                     # iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 3 and 5
                                     ])),
            iaa.contrast.LinearContrast((0.5, 2.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.2),
            iaa.Multiply((0.8, 1.5), per_channel=0.1),
            # iaa.Sometimes(0.7, iaa.Clouds(20)),
            # iaa.Sometimes(0.7, iaa.MultiplyBrightness((1.5, 2.5))),
            iaa.Sometimes(0.7, iaa.Affine(rotate=(-5, 5), shear=(-8, 8))),
            iaa.Sometimes(0.7, iaa.Add((-3, 3), per_channel=0.2)),
            iaa.Sometimes(0.5, iaa.Dropout((0.01, 0.05), per_channel=0.5)),
            iaa.Sometimes(0.9, iaa.OneOf([iaa.imgcorruptlike.Fog(severity=2), iaa.imgcorruptlike.Spatter(severity=2)])),
            iaa.Sometimes(0.3, iaa.Affine(shear=(-3, 3)))], random_order=True)
        seq_det = seq.to_deterministic()


        imageAug    = seq_det.augment_images([plateImg])[0]
        bboxAug     = seq_det.augment_bounding_boxes([bboxAug])[0]
        # bboxAug     = bboxAug.remove_out_of_image().cut_out_of_image()

        bboxAugFormatted = []
        for idx, box in enumerate(bboxAug.bounding_boxes):
            bboxAugFormatted.append((box.x1, box.y1, box.x2, box.y2, plateBoxes[idx][4]))

        return Image.fromarray(imageAug), bboxAugFormatted


    def getStatistics(self):
        return self.statistics

    def nextLine(self):
        if self.isMercosul:
            self.heightRef += 130
            self.widthRef = 80
        else:
            self.heightRef += 130
            self.widthRef = 50


def save_to_csv(file_name, label=False, p1=False, p2=False):
    with open('training.csv', mode='a+') as file:
        f = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not label:
            f.writerow(['UNASSIGNED', file_name + '.jpeg', '', '', '', '', '', '', '', '', ''])
            return
        p1x, p1y = p1
        p2x, p2y = p2
        f.writerow(['UNASSIGNED', file_name + '.jpeg', label, p1x, p1y, '', '', p2x, p2y, '', ''])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        numOfPlates = int(sys.argv[1])
        for _ in range(numOfPlates):
            plateGen = PlateGenerator(showPlates=False, showStatistics=False, contourOnly=False, isMercosul=False, isMotorcycle=True, isRed=True)
            plates = plateGen.generatePlates(numOfPlates=1)
            img = plates[0]['plateImg']
            plate = [plates[0]['plateBoxes'][i][4] for i in range(1,8)]
            chars = [plates[0]['plateBoxes'][i] for i in range(1,8)]
            name = '/home/felipe/Documents/Aiknow/BRLicensePlateGen/generated/red/' + ''.join(plate).lower() + '.jpeg'
            if os.path.isfile(name):
                continue
            img.save(name)
            # w = img.width
            # h = img.height
            # print(plate)
            # for char in chars:
            #     x1 = '{:.4f}'.format(min(max(0, char[0]/w), 1))
            #     y1 = '{:.4f}'.format(min(max(0, char[1]/h), 1))
            #     x2 = '{:.4f}'.format(min(max(0, char[2]/w), 1))
            #     y2 = '{:.4f}'.format(min(max(0, char[3]/h), 1))
            #     save_to_csv('/home/felipe/Documents/Aiknow/BRLicensePlateGen/generated/' + ''.join(plate), label=char[4], p1=(x1, y1), p2=(x2, y2))
    else:
        print("You should specify the number of plates")