from django.http import HttpResponse
from django.shortcuts import render, redirect, render_to_response
from django.template.loader import render_to_string
from django.template import loader, RequestContext
from django.template.response import TemplateResponse


def forStar(request):
	#BARIS = 5
	#for i in range(1, BARIS+1):
#		for j in range(1, i+1):
#			print('%d ' % (i * j), end='')
#		print()
	return render(request,'logic.html')
