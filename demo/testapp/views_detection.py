import cv2
import sys,os
import subprocess

from django.shortcuts import render, redirect, render_to_response
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_page



@login_required(login_url='http://192.168.75.145/login/')
@cache_page(60*60)
def getDetect(request):
	os.system('rm -f /home/bigh/django-cruds-adminlte/demo/testapp/static/upload/frames/* && rm -f /home/bigh/django-cruds-adminlte/demo/testapp/static/upload/hary.* && rm -f /home/bigh/django-cruds-adminlte/demo/testapp/static/upload/later*')
	subprocess.call(['/home/bigh/convert.sh'])
#	cap = cv2.VideoCapture('/home/bigh/django-cruds-adminlte/demo/testapp/static/upload/sources/cars.mp4')
#	car_cascade = cv2.CascadeClassifier('/home/bigh/django-cruds-adminlte/demo/testapp/static/upload/sources/cars.xml')

#	ret, frames = cap.read()

#	count = 0
#	while(cap.isOpened()):
		# membaca frame dari sebuah video
#		ret, frames = cap.read()
		
#		if ret:
			# konversi ke greyscale untuk setiap frame
#			gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
		
			# deteksi car dari berbagai ukuran sebagai input image
#			cars = car_cascade.detectMultiScale(gray, 1.1, 1)

			# menggambar rectangle di setiap car
#			for (x,y,w,h) in cars:
#				cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
		
		# menampilkan frame di setiap window
		#cv2.imshow('video2',frames)	
#			cv2.imwrite("/home/bigh/django-cruds-adminlte/demo/testapp/static/upload/frames/frame-%d.jpg" % count, frames)

		# menunggu ESC untuk perintah STOP
		#if cv2.waitKey(33) == 27:
		#	break
#		count += 1

#	cap.release()
#	cv2.destroyAllWindows()

#	subprocess.call(['/home/bigh/convert.sh'])

	# Dealokasi penggunaan memory
	#cv2.destroyAllWindows()

	return render(request,'motion.html')
