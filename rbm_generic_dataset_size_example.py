# Example of usage: Restricted Boltzmann Machine with continuous-valued outputs
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
# Last modified: Jun, 2018

import imageio
import numpy as np
import os
import torch
from ebm.optimizers import Adam #From the folder "optimizers" it is importing the Adam optimizer
from ebm.samplers import ParallelTempering #Here it is using parallel tempering
from ebm.models import RBM #From the folder "models" it is importing the RBM 

from torchvision import datasets #Imports datasets from torchvision
#import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

#Álvaro lib:
from matplotlib import pyplot
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision
import torchvision.transforms as T
import tempfile
from ebm.quantum_annealing import quantum_annealing



#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd           = 300          # Number of nodes in the hidden layer
learning_rate  = 1e-3         # Learning rate
epochs         = 1            # Training epochs (4 minutes per epoch approx)
k              = 1            # Steps of Contrastive Divergence (CD)
k_reconstruct  = 4000         # Steps of iteration during generation
batch_size     = 32           # Batch size
"""
An H5 file is a data file saved in the Hierarchical Data Format (HDF).
It contains multidimensional arrays of scientific data.
"""
gpu            = True         # Use of GPU

loss=[]                      #y value to plot the well fitting
x_epoch=np.arange(1,epochs+1) #x value to plot the well fitting
validation_values=[]
dataset_size=7              #set the input dataset size
vneurons=dataset_size**2    #input visible neurons in RBM (Only 784 when dataset_size=28)
model_dir      = "RBM"+str(dataset_size)+"x"+str(dataset_size)+".h5"       # Directory for saving last parameters learned
i=1
w_updates2=[]
w_updates=[]
classical=True


#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device
"""
Á: The torch.device enables you to specify the device type
responsible to load a tensor into memory.
The function expects a string argument specifying the device type.
You can even pass an ordinal like the device index. or leave it unspecified
for PyTorch to use the currently available device.
"""
data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.float) 

#Á: Load the MNIST dataset and use it as TRAINING (TENSOR)
#Á: data[1000][400] output: tensor(0.2275)
#Á: A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
#Á Doubt: 0=white 1=black (0,1)= grey
#Á: data.size() output: torch.Size([6000,784])

test = datasets.MNIST('mnist',
                      train=False).test_data.type(torch.float)
#Á: Load the MNIST dataset and use it as TEST (TENSOR)

#If the RBM will be run in QPU for quantum annealing
if (dataset_size!=28):
    data = T.Resize(size=dataset_size)(data) #Á: Now it is ([60000,7,7])
    test = T.Resize(size=dataset_size)(test) #Á: Now it is ([10000,7,7])

# -----------------------------------------------------------------------------
# Show an example of the image that will be introduced into the RBM
# -----------------------------------------------------------------------------
print(data[0].size())
example=data[0]
pyplot.imshow(example, cmap=pyplot.get_cmap('gray'))
#imshow only shows the last imshow (therefore only will be shown the image at the end of the code)
pyplot.savefig("Input_sample"+str(dataset_size)+"x"+str(dataset_size)+".png")

#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

data = (data.view((-1, vneurons)) / 255).to(device) #255 porque lo normaliza de 0 a 1
#Á Doubt: Here it is resizing the number of pixels? Why /255?
test = (test.view((-1, vneurons)) / 255).to(device)
#Á: -1 to automatically resize the tensor, the only condition is to have 784 columns
#Á: these variables send to "device" gpu (cuda)
print(data.size())
print(test.size())
vis = len(data[0]) #Á: 784 is the number of pixels of each "data[i]=number example i"
#Artur: Esta definiendo la capa visible

# According to Hinton this initialization of the visible biases should be
# fine, but some biases diverge in the case of MNIST.
# Actually, this initialization is the inverse of the sigmoid. This is, it
# is the inverse of p = sigm(vbias), so it can be expected that during
# training the weights are close to zero and change little

vbias = torch.log(data.mean(0)/(1 - data.mean(0))).clamp(-20, 20)

#Á: vbias is a tensor of len 784 which is the visible bias b_i
#Á: vbias.size() output: torch.Size([784])
#Á Doubt: clamp returns maximum and minimum values as (-20,20). Why these specifically?

# -----------------------------------------------------------------------------
# Construct RBM
# -----------------------------------------------------------------------------

sampler = ParallelTempering(k=k, #Á:k=1 steps of CD (constrastive divergence)
                            n_chains=100, #Á: Number of parallel chains to be run
                            betas=[0, 0.25, 0.5, 0.75, 1], #Á: List of inverse temperatures of the copies of negative particles
                            #beta: inverse T, aquí lo que hace es enfriar el sistema
                            continuous_output=True) #Á: Optional parameter to output visible activations instead of samples (for continuous-valued functions)

#Á: Here it is using as sampler parallel tempering instead of PCD.
#Á: We will change this to introduce here the sampler of the D-wave QA

optimizer = Adam(learning_rate)
#Á: The optimizer used is the Adam optimizer to the RBM to learn
#Á: type(sampler)=type(optimizer)=type(rbm)=func.

#Á: Define the RBM
rbm = RBM(n_visible=vis,
          n_hidden=hidd,
          sampler=sampler, #Á: Here should be substituted with the QA
          optimizer=optimizer,
          device=device,
          vbias=vbias)

pre_trained = os.path.isfile(model_dir)
#Á: output: False
#Á Doubt: (As expected, the first time it is NOT pre_trained), on next iteration it will be
#Á: model_dir= 'RBM.h5' Directory for saving last parameters learned

if pre_trained: #Á Doubt: For the next iteration when it has the pre_trained
    rbm.load_state_dict(torch.load(model_dir)) #Artur: Si está entrenado, carga los valores del fichero (w,b,h)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

if not pre_trained:
    #Á: If not pre_trained --> Start the training
    validation = data[:10000] #10.000 imágenes (Sergio)
    #Á Doubt: Grab 10.000 images (check with data[:10000][0]). data[:10000][0].size() output: torch.Size([784])
    #Á: Tensor of size([1000, 784]). 10000 rows of 784 columns.
    for _ in range(epochs):
        
        #Á: Training epochs (In the paper is 100 but here 20. 20 epochs=20 min training)
        
        train_loader = torch.utils.data.DataLoader(data,
                                                   batch_size=batch_size, #Á: tamaño del lote
                                                   shuffle=True) #Á: barajar
        
        rbm.train(train_loader, rbm.state_dict(), classical)
        
        """
        w_updates=rbm.train(train_loader)
        print(len(w_updates[0]))
        
        for i in range(len(w_updates)):
            w_updates2.append(w_updates[i])
            
            
        plt.plot(np.arange(len(w_updates2)), w_updates2, 'ro', markersize=0.2)
        plt.title("weight updates 1 hidden node (mean)")
        plt.xlabel("batch number")
        plt.ylabel("w_update")
        plt.locator_params(axis="y", nbins=20)
        plt.legend(['W_update[0].mean(0).item()'])
        plt.savefig("w_update")
        plt.show()
        #Á: Parece una función logaritmica, podría hacer un fit.
        #Á: Vemos que a medida que epoch n (y por ende batch number)
        #Á: tiende a infinito, w_updates2 se aproxima al gradiente de
        #Á: la loss funcion respecto w
        """
        """
        #Á: train the RBM function using the loaded data "train_loader"
        #Á: train_loader: Batch (lote/bunch) of trainig points
        #Á: uses the optimizer stablished, in this example AdaOptimizer (you can change it)
        """
        # A good measure of well-fitting is the free energy difference
        # between some known and unknown instances.
        #print(rbm.free_energy(validation).size())
        #print(rbm.free_energy(test).size())
        gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)

        
        """
        gap_nomean=rbm.free_energy(validation) - rbm.free_energy(test)
        output=rbm.free_energy(validation)
        labels=rbm.free_energy(test)
        """
        #Accuracy:
        
        #accuracy=100*correct/len(train_loader)
        #print("Accuracy = {}".format(accuracy)
        #Hiciese clases para identificar cada número
        """
        rbm.free_energy(validation):
            
            Computes the free energy for a given state of the visible layer.
            validation: unknown instance 
        
        rbm.free_energy(test)
            
            test: known instance
                
        test.size()==validation.size() True
        type(test)==type(validation) True
        
        If they are almost the same, it means that the computed free energy of validation
        is correct, therefore ¿the state computed by the RBM is the correct one?
        
        DOUBT la pregunta
        """
        
        print('Gap = {}'.format(gap.item()))
        loss.append(gap.item())
        print("New loss element added to the list: ", loss)
        
        """
        if(_%5==0):
            print("Plotting epoch: "+str(_)+"...")
            x_axis_test_plot=np.arange(1,len(rbm.free_energy(test))+1)
            plt.scatter(x_axis_test_plot,rbm.free_energy(test),c="black", marker="o", s=0.02)
            plt.title("Free energy for a given state\n of the visible layer")
            plt.xlabel("Image vector. 10.000 total images (test set)")
            plt.ylabel("Free energy (test set)")
            plt.locator_params(axis="y", nbins=20)
            plt.legend(["rbm.free_energy(test)"])
            plt.ylim([-350,-50]) #Set limit for the gif and see how the free energy minimizes
            plt.savefig("Epoch_"+str(_)+"_free_energy_no_mean_applied.png")
            plt.show()
        else:
            continue
        """
            
        #Á: gap.item() element of the tensor, which is a escalar in this situation

    torch.save(rbm.state_dict(), model_dir)
    
    #Á: Save the best parameters learned of the weights and biases in model_dir after the training

    """
    rbm.state_dict():
        
    In PyTorch, the learnable parameters (i.e. weights and biases: w_i, b_i, c_i)
    of a torch.nn.Module model are contained in the model’s parameters
    (accessed with model.parameters()).
    A state_dict is simply a Python dictionary object that maps
    each layer to its parameter tensor.
    
    model_dir: 
        
    Directory for saving last parameters learned 'RBM.h5'
    """


"""
Process doubt:
    
    The training its being done with all the DATA written numbers,
    each iteration is a batch of hand-written (shuffled) numbers
    in which the training is performed for each epoch.
    After each training per epoch (bunch of data) the gap is computed
    and printed to observe how it decreases each iteration.
    
    When the loop finishes, the best rbm.state_dict()
    is saved into the .h5 file.
    
"""

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
"""

print("Plotting the gap\n")

plt.plot(x_epoch, loss, 'r--o')
plt.title("Mean of the difference between\n validation and test free energies")
plt.xlabel("Epoch")
plt.ylabel("Gap")
plt.locator_params(axis="y", nbins=20)
plt.legend(['gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)'])
plt.savefig("gapVSepoch_mean_applied.png")
plt.show()



print("Plotting the LAST validation batch (last iteration training result)\n")

x_axis_validation_plot=np.arange(1,len(rbm.free_energy(validation))+1)
plt.plot(x_axis_validation_plot,rbm.free_energy(validation),"go", markersize=0.2)
plt.title("Validation values per batch\n No applied mean")
plt.xlabel("Datapoint for free energy")
plt.ylabel("Free energy of the validation")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(validation)"])
plt.savefig("Last_Epoch_Validation_Batch_no_mean_applied.png")
plt.show()

print("Plotting the LAST test batch (last iteration training result)\n")

x_axis_test_plot=np.arange(1,len(rbm.free_energy(test))+1)
plt.scatter(x_axis_test_plot,rbm.free_energy(test),c="black", marker="o", s=0.02)
plt.title("Test values per batch\n No applied mean")
plt.xlabel("x datapoint for the test free energy")
plt.ylabel("Free energy of the test")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(test)"])
plt.savefig("Last_Epoch_test_Batch_no_mean_applied.png")
plt.show()

print("Plotting the LAST validation batch difference\n between validation and test")
x_gap_nomean=np.arange(1,len(gap_nomean)+1)
plt.plot(x_gap_nomean,gap_nomean,"bo", markersize=0.2)
plt.title("Difference\n rbm.free_energy(validation) - rbm.free_energy(test) \n in LAST epoch (10.000 images) how the free energy differs")
plt.xlabel("Image number")
plt.ylabel("Free energy of validation-test (no mean)")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(validation) - rbm.free_energy(test)"])
plt.savefig("Last_Epoch_Validation-Test_Batch_no_mean_applied.png")
plt.show()
"""


"""
#All weights of the rbm after training (completely updated after training)
#This can be similarly applied to hbias and vbias after training
rbmweights=[]
for i in range(rbm.weights.size(0)):
    for j in range(rbm.weights.size(1)):
        rbmweights.append(rbm.weights[i][j].item())
plt.plot(np.arange(len(rbmweights)), rbmweights, 'ro', markersize=0.2)
plt.title("All weights of the rbm after training\n (completely updated after training)")
plt.xlabel("weight_i")
plt.ylabel("value")
plt.locator_params(axis="y", nbins=20)
plt.legend(['rbm.weights[i][j].item()'])
plt.savefig("weight")
plt.show()
"""


"""
print("Visualize the real image")
def show_adn_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f,npimg)
show_adn_save("real_image",make_grid(validation.view(batch_size,1,28,28).data)) #antes era -1=32
"""

print('Reconstructing images')
zero = torch.zeros(25, vneurons).to(device)
#Á: Creamos el tensor
images = [zero.cpu().numpy().reshape((5 * dataset_size, 5 * dataset_size))]
#Á: 28x28 1 imagen
#Á: Nuestro gif tiene 25 imágenes (Celdas), por lo tanto, size (5*28,5*28)
for i in range(k_reconstruct):
    #Á: Recuerda: k_reconstruct  = 2000 # Steps of iteration during generation
    zero = sampler.get_h_from_v(zero, rbm.weights, rbm.hbias)
    #Á Doubt: Forward sampling of the hidden layer once the rbm has the weights and hbias trained?
    #Á: Hace una distribución de activación de cuáles de las hidden neurons están activadas (1) o no (0)
    zero = sampler.get_v_from_h(zero, rbm.weights, rbm.vbias)
    #Á Doubt: Backward sampling of the visible layer once the rbm has the weights and vbias trained?
    #Á: Hace una distribución de activación de cuáles de las vissible neurons están activadas (1) o no (0)
    
    #Á: Ahora el sistema tiene vissible y hidden neurons 1 y 0 por nodo. Esta activación representará 1 número.
    #Á: Así estará sampleando las v y h neurons (0,1) durante k_reconstruct veces
    
    if i % 3 == 0:
        datas = zero.data.cpu().numpy().reshape((25, dataset_size, dataset_size))
        #Á: tensor de 25 imágenes, cada una de 28x28
        #Á Doubt: Sin entrenar "zero" es un tensor todo en negro
        #Á Doubt: Pero después de hacer k_reconstruct veces
        #Á: Doubt: "zero" representa un número con sus (v,h) --> (0,1)
        
        image = np.zeros((5 * dataset_size, 5 * dataset_size))
        #Á: Creamos la imagen para rellenar
        for k in range(5):
            for l in range(5):
                image[dataset_size*k:dataset_size*(k+1), dataset_size*l:dataset_size*(l+1)] = datas[k + 5*l, :, :]
                #Á Doubt: Está metiendo los píxeles en cada imagen datas[25,28,28]
        images.append(image)
                #Á Doubt: Guarda el fotograma en images
imageio.mimsave(str(epochs)+"Epochs_RBM_sample"+str(dataset_size)+"x"+str(dataset_size)+".gif", images, duration=0.1)
#Á: Crea un gif con la evolución de los fotogramas a medida que k_reconstruct avanza
#Á: Yo ahora me quedo con el último fotograma para analizarlo
pyplot.imshow(images[-1], cmap=pyplot.get_cmap('gray'))
pyplot.savefig(str(epochs)+"Epochs_Last_photogram"+str(dataset_size)+"x"+str(dataset_size)+".png")

#A: Grab each image of this gif:
image=np.zeros((dataset_size,dataset_size))
image[:,:]=datas[5,:,:]
#Á: El índice i de datas[i,:,:] lo que hace es coger cada sub-imagen, el índice i
#Á: representa la imagen por columnas. i=0 es la primera imagen de arriba de todo
#Á: a la izquierda. i=4 es la imagen de abajo de todo a la izquierda.
#Á: Filas y columnas empiezan a contar con 0. i=5 es columna 1, fila 0 de la
#Á: matriz imagen.



#Á: Uncomment when interested in saving generated images
############################
#Á: Grab and save the images
############################
#generated_img=torch.zeros(25,dataset_size,dataset_size).to(device)
#for i in range(image.size(0)):
#    generated_img[i,:,:]=torch.Tensor(datas[i,:,:])
#for i in range(image.size(0)):
    #save_image(image[i], f"C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/resnet_example/image"+str(i)+"_"+str(epochs)+"epochs_k_reconstruct"+str(k_reconstruct)+"_k_"+str(k)+".png")
#path: C:\Users\Alvar\Downloads\TFM_ebm-torch-master\ebm-torch-master_RBM\Alvaro_Project\SubirGitHub\resnet_example


# -----------------------------------------------------------------------------
# Grab the image and convert it into a tensor
# -----------------------------------------------------------------------------
"""
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision.transforms as T


imagen = Image.open('7_number.png')
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])
#Á: Transform image to tensor
img_tensor = transform(imagen) #Á: Now it is: torch.Size([3, 28, 28])
imagen_resnet=img_tensor[0:1] #Now it is: torch.Size([1, 28, 28])
imagen_resnet = torch.tensor(np.array([imagen_resnet.numpy()]))
#Á: Now it is: torch.Size([1, 1, 28, 28]) to introduce into the resnet
imagen_resnet_input = (imagen_resnet.view((-1, 1,28,28)) / 255).to(device) #Á: Notmalize and introduce into device


pyplot.imshow(imagen, cmap=pyplot.get_cmap('gray'))
"""
############################
#Á: Build the matrix for quantum annealing
############################

"""

from dwave.system import DWaveSampler, EmbeddingComposite
sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})

sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
"""

"""
#Á: This doesn´t work. Sampler needs a dictionary, not a matrix

matrix=torch.zeros(1084,1084).to(device) #300+784=1084, the matrix must be squared 1084x1084
for i in range(len(rbm.state_dict()["vbias"])):
    matrix[i,i]=rbm.state_dict()["vbias"][i].item()
    #Á: Grab each vbias element and introduce in the first diagonal
for i in range(len(rbm.state_dict()["hbias"])):
    matrix[len(rbm.state_dict()["vbias"])+i,len(rbm.state_dict()["vbias"])+i]=rbm.state_dict()["hbias"][i].item()
    #Á: Grab each hbias element and introduce in the continuation of the diagonal
r=0
k=0
t=1
for i in range(300):
    for j in range(784):
        matrix[r][k+t]=rbm.state_dict()["weights"][i][j].item()
        if (i==299 and j==783):
        #Á: Mira, los weights no llenan toda la upper matrix
            print("row of matriz is: ", r)
            print("column of matriz is: ", k+t)
            print("last element of weights is: ", rbm.state_dict()["weights"][i][j].item())
            print("last element of weights in matriz is: ", matrix[r][k+t])
        k=k+1 #Á: Next column in the matrix
        if (k+t==1083): #Á: If we arrive to the last column in matrix
            r=r+1 #Á: Pasa siguiente fila
            k=0   #Á: vuelve a empezar columna
            t=t+1 #Á: empieza en una columna más que la anterior
        else:
            continue
"""

"""

qubit_biases={} #Á: Create dictionary
values=[]
for i in range(len(rbm.state_dict()["vbias"])):
    values.append(rbm.state_dict()["vbias"][i].item()) #Á: Store the value in the list
for i in range(len(rbm.state_dict()["hbias"])):
    values.append(rbm.state_dict()["hbias"][i].item()) #Á: Store the value in the list
    
for i in range(1084):
    qubit_biases[(i,i)]=values[i]   #Create the keys as tuples and assign the value as each value

qubit_weights={} #Á: Create a dictionary
weights=[]
for i in range(300):
    for j in range(784):
        weights.append(rbm.state_dict()["weights"][i][j].item()) #Á: Store all the values in a list
t=1
k=0
for j in range(1084):
    for i in range(1084):
        if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
            break
            break
        qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
        k=k+1
    t=t+1

coupler_strengths=qubit_weights
Q = {**qubit_biases, **coupler_strengths}
sampleset = sampler_auto.sample_qubo(Q, num_reads=100)
#sampleset = sampler_manual.sample_qubo(Q, num_reads=5000)
print(sampleset)

"""


# -----------------------------------------------------------------------------
# ResNet
# -----------------------------------------------------------------------------

from resnet_example.resnet18_generic_dataset_size_example import ResNet, resnet18, model
#Sólo puedo importar clases
#Si lo importo lo ejecuta

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

"""
Comentario:
    
    Es necesario usar una resnet desde scratch y entrenarla
    en vez de una pre-entrenada porque será necesaria para
    reconocer imágenes 7x7 de debido al quantum annealing.
"""


# ------------------------------------------- #
# Evaluate RBM generated images with resnet18 #
# ------------------------------------------- #


"""
# Only uncomment when interested in analyze loaded images from folders
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])

# Read a PIL image

lista=[]
for i in range(25):
    lista.append(transform(Image.open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/resnet_example/image"+str(i)+"_2epochs_k_reconstruct4000_k_4.png")))

imagen_tensor=torch.zeros(25,28,28).to(device)
for i in range(25):
    imagen_tensor[i,:,:]=lista[i][0]
"""    
    
# ---------------------------------------------------------- #
# Grab tensor images from RBM and introduce them into resnet #
# ---------------------------------------------------------- #

tensor_list=[]
for i in range(len(datas)):
    tensor_list.append(torch.Tensor(datas[i,:,:]))
imagen_tensor=torch.zeros(25, dataset_size, dataset_size).to(device)
for i in range(len(datas)):
    imagen_tensor[i,:,:]=tensor_list[i]
imagen_tensor_input=(imagen_tensor.view((-1, 1, dataset_size,dataset_size)) / 255).to(device)
    
#imagen_tensor_input = (imagen_tensor.view((-1, 1,28,28)) / 255).to(device)
results=[]
for i in range(len(datas)):
    logits,probas=model(imagen_tensor_input.to(device)[i,None])
    for j in range(10):
        print("prob del número ", str(j), " es: ", probas[0][j]*100)
        if (probas[0][j]*100>95):
            results.append(j)
            print("Number found! Number is: "+str(j)+"\n")
        else:
            continue
    print("Next image to analyze: \n")

#Á: Test resnet

misclassified=[]
accuracy=len(results)/25
print("accuracy of this set is: ", accuracy)
#Á: accuracy is the ones that have prob>0.95 divided by the total dataset