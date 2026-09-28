# Converting EEGs into correlation matrices
# Author: Anna Calissano
# trial_number: 
# sensor_position: 64 nodes
# sample_number:
# value: actual signal value
library(data.table)
library(dplyr)
library(archive)
library(igraph)
setwd("~/Datasets/eeg_full")
dir_save="~/Datasets/eeg_full_correlations/"
# Preprocessing
files_1=list.files()
for(i in ii:length(files_1)){
  # the 4th letter is alcoholic or not alcoholic
  type_patient=substr(files_1[i], 4, 4)
  code_patient=substr(files_1[i], 9, 11)
  
  # untar 
  files_2=untar(files_1[i],list=TRUE)
  untar(files_1[i])
  for(j in 2:length(files_2)){
    if(files_2[j]!= "co2c1000367/co2c1000367.rd.004.gz"){
      data=read.table(gzfile(files_2[j]),col.names = c('trial_number', 'sensor_position', 'sample_number', 'value'))  
      
      info_simul <- gsub("[[:punct:],[:space:]]",'',readLines(gzfile(files_2[j]))[4])
      
      corr_vect=data.frame(t(combn(unique(data$sensor_position), 2, function(i)
        list(v1 = i[[1]], 
             v2 = i[[2]], 
             value = cor(data$value[data$sensor_position %in% i[[1]]], 
                         data$value[data$sensor_position %in% i[[2]]])))))
      
      corr_vect$X1 =as.character(corr_vect$X1)
      corr_vect$X2 =as.character(corr_vect$X2)
      corr_vect$X3 =as.numeric(corr_vect$X3)
      
      colnames(corr_vect) = c('sensor1','sensor2','weight')
      write.table(corr_vect,file=paste(dir_save,paste(type_patient,code_patient,info_simul,'.csv',sep = ''),sep = ''))
      rm(corr_vect,data,info_simul)
      }
  }
  rm(type_patient,code_patient,files_2)
  }
