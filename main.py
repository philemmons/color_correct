from PIL import Image
import numpy as np
import cv2

def minMax( value ):#The adjusted image's color range were out of bounds.
    if value> 255:
        return 255
    if value< 0:
        return 0
    else: return int(value)
''' 
**future work**   
#def normalizeMinMax( value ):
http://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
  import numpy as np

x = np.array([[1000,  10,   0.5],
              [ 765,   5,  0.35],
              [ 800,   7,  0.09]])

x_normed = x / x.max(axis=0)

print(x_normed)
# [[ 1.     1.     1.   ]
#  [ 0.765  0.5    0.7  ]
#  [ 0.8    0.7    0.18 ]]  
'''
#Each RGB band image is multiplied by it corresponding color blind factor.
#The saved files are used by opencv.      
def pixelLand( wide, tall, colorImRGB, filename, colorMult, row):
    
    for x in range( 0, wide ):
        for y in range( 0, tall):
            r_rpix, g_rpix, b_rpix = colorImRGB.getpixel((x,y))
            rr = minMax(r_rpix * colorMult[row][0])
            gr = minMax(g_rpix * colorMult[row][1])
            br = minMax(b_rpix * colorMult[row][2])
            colorImRGB.putpixel((x,y), (rr,gr,br))
        
    colorImRGB.save(filename) 



#This is a check to see if the correct image is being processed.
def Normal():
    imCV = cv2.imread('1a.jpg')
    cv2.imshow('Normal',imCV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
''' The four converters use the same format, which could be written into single method in the future.'''    
def Prota():
#color matrix
    Protanopia= [ [.56667, .43333, 0], 
                  [.55833, .44167, 0], 
                  [0, .24167, .75833] ]
#Converting the single band into three bands to be altered.
    rimRGB = Image.open("rim.jpg").convert('RGB')
    gimRGB = Image.open("gim.jpg").convert('RGB')
    bimRGB = Image.open("bim.jpg").convert('RGB')
    
#Apply the matrix to the image.
    pixelLand(width, height, rimRGB, 'rimP.jpg', Protanopia, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', Protanopia, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', Protanopia, 2)
    
#openCV has features for transparency and merging two images at a time.
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)
#Alpha is the brightness - range: -3 to -3
    alpha = float(2.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))
#Result to console and save file to be uploaded.
    cv2.imshow('Red-Blind/Protanopia',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('Protanopia.jpg', new_img)
        
def Deute():#To much green...
    Deuteranopia= [ [.625, .375, 0], 
                  [.70, .30, 0], 
                  [0, .30, .70] ]
    
    rimRGB = Image.open("rim.jpg").convert('RGB')
    gimRGB = Image.open("gim.jpg").convert('RGB')
    bimRGB = Image.open("bim.jpg").convert('RGB')

    pixelLand(width, height, rimRGB, 'rimP.jpg', Deuteranopia, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', Deuteranopia, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', Deuteranopia, 2)
    
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)

    alpha = float(2.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))

    cv2.imshow('Green-Blind/Deuteranopia',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('Deuteranopia.jpg', new_img)
   
def Trita():
    Tritanopia= [ [.95, .05, 0], 
                  [0, .43333, .56667], 
                  [0, .475, .525] ]
    
    rimRGB = Image.open("rim.jpg").convert('RGB')
    gimRGB = Image.open("gim.jpg").convert('RGB')
    bimRGB = Image.open("bim.jpg").convert('RGB')

    pixelLand(width, height, rimRGB, 'rimP.jpg', Tritanopia, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', Tritanopia, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', Tritanopia, 2)
    
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)

    alpha = float(2.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))

    cv2.imshow('Blue-Blind/Tritanopia',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('Tritanopia.jpg', new_img)
 
def Achro():#To much green....
    #This image should be slightly brighter than a stock greyscale.
    imCV = cv2.imread('1a.jpg')
    imACH = cv2.cvtColor(imCV, cv2.COLOR_BGR2GRAY)
    '''
    Achromatopsia= [ [.299, .587, .114], 
                     [.299, .587, .114], 
                     [.299, .587, .114] ]
    
    rimRGB = Image.open("rim.jpg").convert('RGB')
    gimRGB = Image.open("gim.jpg").convert('RGB')
    bimRGB = Image.open("bim.jpg").convert('RGB')

    pixelLand(width, height, rimRGB, 'rimP.jpg', Achromatopsia, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', Achromatopsia, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', Achromatopsia, 2)
    
#pil to cv2
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
#the green values are to high    
    cv2.imshow('Monochromacy',rimCV)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)

    alpha = float(2.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))
    '''
    cv2.imshow('Monochromacy',imACH)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('Achromatopsia.jpg', imACH)

'''Attempt to reverse the process, step by step.
   The following three functions do the same thing.'''   
def AdjProta():
#This value will be used to darken the image.
    alpha = float(0.5)
    AdjPro= [ [1.78, 2.72, 0], 
              [1.79, 2.72, 0], 
              [0, 4.12, 1.32] ]
#RGB band images are 
    bwList = []
#This file is the color blind version and darken it.    
    proCV = cv2.imread("Protanopia.jpg")
    imCV = cv2.multiply(proCV, np.array([alpha]))
    cv2.imwrite('improCV.jpg', imCV)
#Pil split to the resuce.
    imPRO = Image.open('improCV.jpg')
    width, height = imPRO.size

#Split the image for the second time into RGB channels to be processed.
    rimPRO, gimPRO, bimPRO = imPRO.split()
    bwList.extend( (rimPRO, gimPRO, bimPRO) )
    
    for i in range( 0,3 ):
        bwList[i].convert('RGB')
        bwList[i].save( str(i)+"PRO.jpg")
#Convert from band L to RGB    
    rimRGB = Image.open("0PRO.jpg").convert('RGB')
    gimRGB = Image.open("1PRO.jpg").convert('RGB')
    bimRGB = Image.open("2PRO.jpg").convert('RGB')
    
#The range of color adjust is way off.
    pixelLand(width, height, rimRGB, 'rimP.jpg', AdjPro, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', AdjPro, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', AdjPro, 2)
#Images are altered and saved.   
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
#'Merged into a single image.    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)
#Brightened and this is were the image contrast would occur.
    alpha = float(1.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))
#Result displayed and saved.
    cv2.imshow('Adjusted Protanopia',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('uploadMe.jpg', new_img)
   
def AdjDeute():
    alpha = float(0.5)
    AdjDeu= [ [1.6, 2.67, 0 ], 
              [1.43, 3.33, 0], 
              [0, 3.33, 1.43] ]
    bwList = []
    
    deuCV = cv2.imread("Deuteranopia.jpg")
    imCV = cv2.multiply(deuCV, np.array([alpha]))
    cv2.imwrite('impdeuCV.jpg', imCV)
    
    imDEU = Image.open('impdeuCV.jpg')
    width, height = imDEU.size

    #split the image into RGB channels
    rimDEU, gimDEU, bimDEU = imDEU.split()
    bwList.extend( (rimDEU, gimDEU, bimDEU) )
    
    for i in range( 0,3 ):
        bwList[i].convert('RGB')
        bwList[i].save( str(i)+"DEU.jpg")
    
    rimRGB = Image.open("0DEU.jpg").convert('RGB')
    gimRGB = Image.open("1DEU.jpg").convert('RGB')
    bimRGB = Image.open("2DEU.jpg").convert('RGB')
    
    #****** the range of color adjust is way off  ***
    
    pixelLand(width, height, rimRGB, 'rimP.jpg', AdjDeu, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', AdjDeu, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', AdjDeu, 2)
    
#pil to cv2
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)

    alpha = float(1.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))

    cv2.imshow('Adjusted Deuteranopia',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('uploadMe.jpg', new_img)

def AdjTrita():
    alpha = float(0.5)
    AdjTri= [ [1.78, 2.72, 0], 
              [1.79, 2.72, 0], 
              [0, 4.12, 1.32] ]
    bwList = []
    
    triCV = cv2.imread("Tritanopia.jpg")
    imCV = cv2.multiply(triCV, np.array([alpha]))
    cv2.imwrite('imtriCV.jpg', imCV)
    
    imTRI = Image.open('imtriCV.jpg')
    width, height = imTRI.size

    #split the image into RGB channels
    rimTRI, gimTRI, bimTRI = imTRI.split()
    bwList.extend( (rimTRI, gimTRI, bimTRI) )
    
    for i in range( 0,3 ):
        bwList[i].convert('RGB')
        bwList[i].save( str(i)+"TRI.jpg")
    
    rimRGB = Image.open("0TRI.jpg").convert('RGB')
    gimRGB = Image.open("1TRI.jpg").convert('RGB')
    bimRGB = Image.open("2TRI.jpg").convert('RGB')
    
    #****** the range of color adjust is way off  ***
    
    pixelLand(width, height, rimRGB, 'rimP.jpg', AdjTri, 0)
    pixelLand(width, height, gimRGB, 'gimP.jpg', AdjTri, 1)
    pixelLand(width, height, bimRGB, 'bimP.jpg', AdjTri, 2)
    
#pil to cv2
    rimCV = cv2.imread('rimP.jpg')
    gimCV = cv2.imread('gimP.jpg')
    bimCV = cv2.imread('bimP.jpg')
    
    dst2 = cv2.addWeighted(rimCV,.6,gimCV,.6,0)
    dst1 = cv2.addWeighted(dst2,.6,bimCV,.6,0)

    alpha = float(1.0)
    new_img = cv2.multiply(dst1,np.array([alpha]))

    cv2.imshow('new_image',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    cv2.imwrite('uploadMe.jpg', new_img)
    
def AdjAchro():
    #greyscale - [.3,.59,.11]
    Normal()    

def BLAH(): #replace this before uploading - indent also
#This presents the programs features to the user.
#Also, the values are reset after every selection.
#Rinse and repeat.
    userInput =0
    while(userInput != 10):
        print("Please choose one of the following: ")
        print("1 - Normal Color Vision")
        print("2 - Red-Blind/Protanopia image")
        print("3 - Green-Blind/Deuteranopia image")
        print("4 - Blue-Blind/Tritanopia image")
        print("5 - Monochromacy/Achromatopsia image")
        print("6 - Adjusted Protanopia image")
        print("7 - Adjusted Deuteranopia")
        print("8 - Adjusted Tritanopia image")
        print("9 - Adjusted Achromatopsia image")
        print("10 - Exit image")
        
        userInput = int(input("Enter your choice: "))
        if (0< userInput) and (userInput <10):
            
            #imRGB = Image.open("1a.jpg")       
            #width, height = imRGB.size
            
            rgbLIST =[]
    #First divide of the image into RGB channels as to be split in the calling function.
            rim, gim, bim = imRGB.split()
            rgbLIST.extend( (rim, gim, bim) )
            
            for i in range( 0,3 ):
                rgbLIST[i].convert('RGB')
               
            rgbLIST[0].save("rim.jpg")
            rgbLIST[1].save("gim.jpg")
            rgbLIST[2].save("bim.jpg")
        
            UserSelect[userInput]()
            
'''****MAIN****'''            
#Python's version of a switch-case to be used in the menu selection.
UserSelect = {
    1:Normal,
    2:Prota,
    3:Deute,
    4:Trita,
    5:Achro,
    6:AdjProta,
    7:AdjDeute,
    8:AdjTrita,
    9:AdjAchro
    }

'''Moved these two lines from the BLAH function.
   This will make the program callable from CLI.'''
imRGB = Image.open("1a.jpg")       
width, height = imRGB.size           
BLAH()
#EOF
