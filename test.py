import random
def formerCarres(tailleBloc,imArray):
  FORME_IMAGE=imArray.shape

  if(FORME_IMAGE[0]%tailleBloc!=0 or FORME_IMAGE[1]%tailleBloc!=0):
    print("erreur sur la dimension de l' image les blocs vont pas fit")
  blocs=[]
  for i in range(FORME_IMAGE[0]//tailleBloc):
    for j in range(FORME_IMAGE[1]//tailleBloc):
      vector=[]
      x=tailleBloc*i
      y=tailleBloc*j
      for k in range(tailleBloc):
        for l in range(tailleBloc):
          vector.append(imArray[x+k][y+l])

      blocs.append(np.array(vector))

  return np.array(blocs)



def formerCarresCouleur(tailleBloc,colArray):

  FORME_IMAGE=colArray.shape
  if(FORME_IMAGE[0]%tailleBloc!=0 or FORME_IMAGE[1]%tailleBloc!=0):
    print("erreur sur la dimension de l' image les blocs vont pas fit")
  blocs=[]
  for i in range(FORME_IMAGE[0]//tailleBloc):
    for j in range(FORME_IMAGE[1]//tailleBloc):
      vector=[]
      x=tailleBloc*i
      y=tailleBloc*j
      for k in range(tailleBloc):
        for l in range(tailleBloc):
          vector.append(colArray[x+k][y+l][0])
          vector.append(colArray[x+k][y+l][1])
          vector.append(colArray[x+k][y+l][2])

      blocs.append(np.array(vector))

  return np.array(blocs)




def initialisation(vecteurs):

  vecteurs = np.array(vecteurs)
  C00 = np.mean(vecteurs,axis=0)

  return np.array([C00])


def creerEpsilon(dimension):

    vector=[]
    for i in range(dimension):
        vector.append(random.random())
    return np.array(vector)


import numba
from numba import njit


@njit
def closest_in_codebook(vector_data, codebook):
    dist = np.linalg.norm(vector_data - codebook[0])
    index = 0

    for i in range(len(codebook)):
        dist_courante = np.linalg.norm(vector_data - codebook[i])
        if dist_courante < dist:
            dist = dist_courante
            index = i

    return index


   



def calculateDistorsion(vecteurs,codebook):
    sum=0
    for i in range(len(vecteurs)):
       sum+=np.linalg.norm(vecteurs[i]-codebook[closest_in_codebook(vecteurs[i],codebook)])**2
    return sum/len(vecteurs)
  


    




import math
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import random




















def LBGalgorithmOnCustomDataAndDraw(data,seuilStop,numberofCodebookVectors,renderDraw=False):
    #data has to be a list of vectors

    uniquedrawid=0
    def draw(drawid):
        if(renderDraw==False):
            return
        plt.figure()
        x = [vector[0] for vector in data]
        y = [vector[1] for vector in data]

        # Create the scatter plot
        plt.scatter(x, y,color='red')
        #print(codebook)
        xx = [vector[0] for vector in codebook]
        yy = [vector[1] for vector in codebook]

        # Create the scatter plot
        plt.scatter(xx, yy,color='blue')

        #save the figure as a png named iteration number.jpg
        plt.savefig(str(len(codebook))+"_"+"plot"+str(drawid)+".png")
        
        


    blockVectors=data
    codebook=initialisation(blockVectors)

    
    
    
    while len(codebook)<numberofCodebookVectors:
        print("the current codebook length is"+str(len(codebook)))


        #save current codebook to a file named save{len(codebook)}.txt
        np.savetxt("save"+str(len(codebook))+".txt",codebook)
        print("the current codebook is"+str(codebook))


        counter=0
        
        #create pyplot and plot all the vectors in data (which are 2d) and all the 2d vectors in codebook
        
        #print("the current codebook[0] is"+str(codebook[0]))
        #print("the current codebook length is"+str(len(codebook)))
        #print(currentDistorsion-oldDistorsion)
        #split in two each vector from the codebook
        eps=creerEpsilon(len(codebook[0]))
        newCodebook=[]
        draw(uniquedrawid)
        uniquedrawid+=1
        for c in codebook:
            newCodebook.append(c+eps)
            newCodebook.append(c-eps)
        codebook=np.array(newCodebook)

        #update each splitted vectors position in a loop while the distorsion can decrease (calculate a mean of the vectors close to them)
        currentDistorsion=calculateDistorsion(blockVectors,codebook)
        oldDistorsion=currentDistorsion+2*seuilStop # dont stop at the first iteration
        while(abs(oldDistorsion-currentDistorsion)>seuilStop):
            
            counter+=1
            draw(uniquedrawid)
            uniquedrawid+=1
            print(currentDistorsion-oldDistorsion)
            print("is the current distorsion-oldDistorsion")
            
            closestInCodebookArray=[closest_in_codebook(blockVectors[i],codebook) for i in range(len(blockVectors))]
            #for each vector in our data, this array contains the index of the closest vector in the codebook


            
            newCodebook=[]
            for i in range(len(codebook)):
                interestingIndexes=np.where(np.array(closestInCodebookArray)==i)


                

                try:
                   #mean over these vectors
                    average=np.mean(blockVectors[interestingIndexes],axis=0)
                except:
                    average=codebook[i]
                    print("EXCEPTIONN")

                ##print("average shape:"+str(average.shape))
                newCodebook.append(average)
            codebook=np.array(newCodebook)

            
        
        


            #update oldDistorsion and currentDistorsion
            oldDistorsion=currentDistorsion
            currentDistorsion=calculateDistorsion(blockVectors,codebook)
            #print("ending while loop")
    #print("the final codebook length is"+str(len(codebook)))
    draw(uniquedrawid)
    np.savetxt("save"+str(len(codebook))+".txt",codebook)




def importCodebookFromTxt(filename):
    codebook=np.loadtxt(filename, delimiter=" ")
    return codebook





def recreate_image(blocks, block_size=4, img_size=32):
    num_blocks = img_size // block_size
    img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    block_idx = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = np.array(blocks[block_idx]).reshape(block_size, block_size, 3)
            img_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
            block_idx += 1

    return img_array



def quantizeImage(codebook,originalImg,size,finalSize=32):
    #take an image and replace it block by block with the codebook
    #create the blocks

    blocks=formerCarresCouleur(size,originalImg)

    print(blocks[0])
    print(blocks[900])
    print(blocks[1901])

    #quantize each block
    for i in range(len(blocks)):
        blocks[i]=codebook[closest_in_codebook(blocks[i],codebook)]

        # if(random.randint(0,100)==0):
        #     print("the current block is"+str(blocks[i]))
        
           
    
    #recreate the image
    image=recreate_image(blocks,size,finalSize)
    return image





def mainCode():
    # image = Image.open('lena.png')

    # image_gray=image.convert("L")
    # grayArray = np.asarray(image_gray)

    # TAILLE_BLOC = 8

    # LBGalgorithm(grayArray,TAILLE_BLOC,1,8)




    
    


    def renderPlots():
        num_points = 30
        num_centroids = 4

        # Define the centroids
        centroids = np.array([[0, 0], [8, 8], [0, 8], [8, 0]])

        # Generate random points around the centroids
        points = []
        for centroid in centroids:
            #change random seed for np
            
            points.extend(1.2*np.random.randn(num_points // num_centroids, 2) + centroid)

        visuarray = np.array(points)

        

        #delete all files in the folder than end with =.png
        import os
        import glob
        files = glob.glob('*.png')
        for f in files:
            os.remove(f)
        LBGalgorithmOnCustomDataAndDraw(visuarray,2,8,renderDraw=True)



    from tensorflow.keras.datasets import cifar10

        # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()



    def trainOnCifar10(set):
        #saves to a txt files the different codebooks
        #dleete all files in the folder than end with =.txt
        import os
        import glob
        files = glob.glob('save*.txt')
        for f in files:
            os.remove(f)
        
        

        # Access and use the loaded dataset
        # Example: Print the shapes of the datasets
        X_train=set[:100]


        


        trainingBlocs=[]

        progress=0
        for i in X_train:
            blocsForImage=formerCarresCouleur(2,i)
            trainingBlocs.extend(blocsForImage)
            progress+=1

            if(progress%200==0):
                print("progress:"+str(progress))

        trainingBlocs=np.array(trainingBlocs)
        #get info on this array
        print(trainingBlocs.shape)



        LBGalgorithmOnCustomDataAndDraw(trainingBlocs,100,1024)



    #trainOnCifar10(X_train)

    def openFromSave():
        codebook=importCodebookFromTxt("save1024.txt")
        print(codebook.shape)
        original=X_train[0]
       # img=quantizeImage(codebook,original,2)

        arrayLena=plt.imread("lena.png")
        arrayLena=255*arrayLena
        #arrayLena=X_train[0]
        print(arrayLena.shape)
        


        lena=quantizeImage(codebook,arrayLena,2,finalSize=512)
        print(lena.shape)

        #print sample numbers in lena array
        print(lena[0][0])
        print(lena[0][1])
        print(lena[0][2])
        plt.imsave("lenaQuant.png",lena)



        print(img)
        print(img.shape)
        #save img as png
        plt.imsave("img.png",img)

        plt.imsave("original.png",original)
        num_images = 10
        #create an empty big_image

        image_size = 32
        big_image = np.zeros((image_size*num_images, image_size*2, 3), dtype=np.uint8)

        for i in range(num_images):
            original = X_train[i]
            #img = quantizeImage(codebook, original,2)

            # Fill the big image with the original and reconstructed images
            big_image[i*image_size:(i+1)*image_size, :image_size] = original
            big_image[i*image_size:(i+1)*image_size, image_size:] = img

            # # Save the original and reconstructed images as PNG files
            # plt.imsave(f"original{i}.png", original)
            # plt.imsave(f"img{i}.png", img)

        # Display the big image using matplotlib
        plt.imsave("big_image.png",big_image)


        #display the quantized 2x2 blocks
        #recreate_image(blocks, block_size=4, img_size=32):

        imgOfCodebook=recreate_image(codebook,2,64)
        plt.imsave("codebook.png",imgOfCodebook)
    openFromSave()




mainCode()
