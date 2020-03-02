from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, render_to_response

@login_required(login_url='http://192.168.75.145/login/')
def getgmPlot(request):
	return render_to_response('gmplot.html')
