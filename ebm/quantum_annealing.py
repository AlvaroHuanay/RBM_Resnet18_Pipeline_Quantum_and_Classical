# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:46:27 2022

@author: Alvar

This is the code for quantum annealing. This fragment creates the Q
matrix, uses automated embedding and samples.

Falta: Calcular avg_model de weights, vbias, hbias y return a la rbm

Nota (elimina esta nota una vez montes toda la pipeline)
Para enseñarle a Artur cómo funciona, ejecuta primero la RBM
rbm_example.py y en la terminal escribe estos comandos (escribe
lo que está dentro de la función def)
"""

#libraries:
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import torch
import time


def quantum_annealing(rbm_state_dict,batch_num , qubo):
    #qubo=bool para saber si queremos hacer el sampling con qubo model o ising model
    
    #rowscolumns=rbm.state_dict()["vbias"].size(0)+rbm.state_dict()["hbias"].size(0)

    vbias=rbm_state_dict["vbias"]
    hbias=rbm_state_dict["hbias"]
    weights_tensor=rbm_state_dict["weights"]
    
    # -------------- #
    # qubit biases   #\
    # -------------- #
    
    qubit_biases={} #Á: Create dictionary
    values=[]
    for i in range(len(vbias)):
        values.append(vbias[i].item()) #Á: Store the value in the list
    for i in range(len(hbias)):
        values.append(hbias[i].item()) #Á: Store the value in the list
    
    for i in range(vbias.size(0)+hbias.size(0)):
        qubit_biases[(i,i)]=values[i]   #Create the keys as tuples and assign the value as each value
    
    # ----------------- # 
    # coupler_strengths #
    # ----------------- #
    
    qubit_weights={} #Á: Create a dictionary
    weights=[]
    k=0
    for i in range(len(hbias)):
        for j in range(len(vbias)):
            weights.append(weights_tensor[i][j].item()) #Á: Store all the values in a list
            
    for j in range(len(vbias)): #para cada fila de la matriz
        for i in range(len(vbias)): #para cada columna de la matriz
            qubit_weights[(j,i+len(vbias))]=weights[k] #Á: Add the element list to the dictionary
            k=k+1
    
    """
    t=1
    k=0
    #matriz 1084x1084
    for j in range(1084): #para cada fila de la matriz
        for i in range(1084): #para cada columna de la matriz
            if (i+t==1084): #si llega a la última columna de la matriz
                break #exit i loop to add +1 to j y t para pasar a la siguiente fila por encima de la diagonal
            qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
            k=k+1
            if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
                print("Se acabaron los elementos de weights, quedarán huecos.\n Último elemento del diccionario es fila "+str(j)+"columna "+str(i+t))
                break
        if (k==len(weights)):
            break
        t=t+1
    """
    
    """
    
    
    t=1
    k=0
    for j in range(rowscolumns):
        for i in range(rowscolumns):
            if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
                break
                break
            qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
            k=k+1
        t=t+1
    """
    
    # ------------------------------------------- #
    # Build the Q matrix, sample and compute avgs #
    # ------------------------------------------- #
    
    coupler_strengths=qubit_weights
    Q = {**qubit_biases, **coupler_strengths}
    num_reads=100
    
    #sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'}, token="DEV-0467747fa9018fbad8c93ccdc0eab18075f8e5bf")
    sampler_auto = EmbeddingComposite(DWaveSampler( solver={'qpu': True, 'topology__type': 'chimera'}, token="DEV-b69802185fb2cc7060a2c4615e3d3155706d69ae"))
    
    
    """
    #Find the index of the minimum energy in the Ising model:
        
    ising_df.iloc[ising_df["energy"].idxmin()]
    #Los valores de la diagonal son los vbias y hbias. Han de ser multiplicaciones +1*-1 y sus permutaciones
    #La configuración vbias hbias que me da el mínimo es la correcta para ese annealing.
    #5000 samplings, 5000 energías, cojo la mínima energía, y la configuración vbias + hbias y hago su avg product.
    """
    
    #avgs
    
    if (qubo==True):
        
    # -------------------- #
    # avgs with QUBO model #
    # -------------------- #
        
    
        #Qubo model
        print("Sampling...\n")
        start_sampling = time.time()
        sampleset_qubo = sampler_auto.sample_qubo(Q, num_reads=num_reads) #Á: Auto sampler mode
        end_sampling = time.time()
        print("Dwave process to send + sample + receive the data is: ", end_sampling-start_sampling)
        #sampleset = sampler_manual.sample_qubo(Q, num_reads=5000) #Á: Manual sampler mode
        #print(sampleset_qubo)
        #print("Creating pandas dataframe...\n")
        qubo_df=sampleset_qubo.to_pandas_dataframe()
        #print("Plotting energy per sample in the annealing QUBO model...\n")
        #plt.title("Energy per sample in the annealing Qubo model")
        #plt.xlabel("Sample number")
        #plt.ylabel("Energy of the sample")
        #qubo_df["energy"].plot(c="black", marker="o", linewidth=0.02, markersize=0.05);
        
        print("Computing negative phases...\n")
        #start_negphase=timer()
        
        #v_model:
            
        #v_df=torch.zeros(vbias.size(0),num_reads) #Á: Creamos un dataframe con toda la información de vbias necesaria
        v_model=torch.zeros(vbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(v_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            v_model[j]=1/num_reads*np.cumsum(qubo_df.iloc[:,j]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
                #Á: Este será v_model, la negative phase del visible bias. Donde el gradiente de la loss: update=vpos-vneg. v_model=veng
        print(v_model)
        #h_model:
        """
        h_df=torch.zeros(hbias.size(0),num_reads)
        h_model=torch.zeros(hbias.size(0))
        for j in range(h_df.size(0)):
            for i in range(h_df.size(1)):
                h_df[j][i]=qubo_df.iloc[i,j+vbias.size(0)]
            h_df[j]=1/h_df.size(1)*np.cumsum(h_df[j])
            h_model[j]=h_df[j][-1]
        """
        
        h_model=torch.zeros(hbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(h_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            h_model[j]=1/num_reads*np.cumsum(qubo_df.iloc[:,j+v_model.size(0)]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)

        print(h_model)
        #w_model:
            
        
        w_model=torch.zeros(weights_tensor.size(0),weights_tensor.size(1))
        #300*49 avg elements que vamos a calcular para la matriz
        for j in range(weights_tensor.size(0)):
            for i in range(weights_tensor.size(1)):
                elements_sum=qubo_df.iloc[:,i]*qubo_df.iloc[:,j+weights_tensor.size(1)]
                avg_element=1/num_reads*np.cumsum(elements_sum)
                w_model[j][i]=avg_element.iloc[-1]
    
        print(w_model)
        #end_negphase=timer()
        #print("Negative phases computed. Runtime: \n", end_negphase-start_negphase)
        
    # ---------------------- #
    # avgs with Ising  model #
    # ---------------------- #
    
    else:
        
        #Ising model
    
        sampleset_ising= sampler_auto.sample_ising(qubit_biases, coupler_strengths, num_reads=num_reads) #Á: Auto sampler mode
        print(sampleset_ising)
        ising_df=sampleset_ising.to_pandas_dataframe()
        plt.title("Energy per sample in the annealing Ising model")
        plt.xlabel("Sample number")
        plt.ylabel("Energy of the sample")
        ising_df["energy"].plot(c="blue", marker="o", linewidth=0.02, markersize=0.05);
        
        #v_model:
            
        v_df=torch.zeros(vbias.size(0),num_reads) #Á: Creamos un dataframe con toda la información de vbias necesaria
        v_model=torch.zeros(vbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(v_df.size(0)): #Á: Para cada fila del dataframe (49)
            for i in range(v_df.size(1)): #Á: Para cada columna del dataframe (5000)
                v_df[j][i]=ising_df.iloc[i,j] #Á: Coge cada elemento del Ising sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            v_df[j]=1/v_df.size(1)*np.cumsum(v_df[j]) #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
            v_model[j]=v_df[j][-1] #Á: Este será v_model, la negative phase del visible bias. Donde el gradiente de la loss: update=vpos-vneg. v_model=veng
        
        #h_model:
            
        h_df=torch.zeros(hbias.size(0),num_reads)
        h_model=torch.zeros(hbias.size(0))
        for j in range(h_df.size(0)):
            for i in range(h_df.size(1)):
                h_df[j][i]=ising_df.iloc[i,j+vbias.size(0)]
            h_df[j]=1/h_df.size(1)*np.cumsum(h_df[j])
            h_model[j]=h_df[j][-1]
    
        #w_model:
        
        w_model=torch.zeros(weights_tensor.size(0),weights_tensor.size(1))
        #300*49 avg elements que vamos a calcular para la matriz
        for j in range(weights_tensor.size(0)):
            for i in range(weights_tensor.size(1)):
                elements_sum=ising_df.iloc[:,i]*ising_df.iloc[:,j+weights_tensor.size(1)]
                avg_element=1/num_reads*np.cumsum(elements_sum)
                w_model[j][i]=avg_element.iloc[-1]

    return v_model, h_model, w_model
