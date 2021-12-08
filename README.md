# Covid-Detection-Using-Chest-X-Rays-Flask-App
Covid Detection Using Chest X-Rays with the help of CNN
<br>
<div>
Infected patients show common signs include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, the infection can cause pneumonia, severe acute respiratory syndrome, kidney failure, and even death.
Patients with respiratory symptoms are advised to stay isolated and undergo further clinical examination RT-PCR (Reverse transcription-polymerase chain reaction). PCR testing is the medical standard to identify COVID-19. However, it might take hours to receive results.
With the growing number of cases walking into hospitals, alternatively, chest X-ray is used as the initial element to review the clinical situation of a patient. Our model will help the radiology experts to detect patterns of Covid-19 at a much faster rate.
<br><br>
  
The X-ray findings that strongly suspect of dealing with COVID-19 infection are the ground glass patterned area, which affects both lungs, in particular the lower lobes, and especially the posterior segments, with a fundamentally peripheral and subpleural distribution in initial stages.
The model shows whether the person is infected with Covid or he is Normal along with providing the percentage for each case. 
Based on the output of the model:
<br>
1) If the X-ray shows any pathological findings, patients are admitted for further diagnosis. 
2) If the X-Ray is normal, patients are requested to go home and wait for PCR test results.
<br>

 <br>
Our CNN Model is built using ResNet-50 architecture. ResNet-50 is a convolutional neural network that is 50 layers deep.<br> 
ResNet50 is a variant of ResNet model which has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer.
<br>
  </div>

Web Application Deployed on Heroku: https://covid-setu-spit.herokuapp.com/
<br>
Upload PNG images of the X-Rays for COVID-19 detection <br>
Colab Notebook link :
https://colab.research.google.com/drive/1vroZTrxHMkexEzknPK6VOIGGiRels1-o?usp=sharing
<br>
Kaggle Dataset Link:
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database



To run the app in your local environment run the requirements from the requirements.txt file. <br>
Then <br>
run python app.py
