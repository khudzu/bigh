# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_page
from django.contrib.auth import login,authenticate
from django.shortcuts import render, redirect, render_to_response
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from testapp.tokens import account_activation_token
from django.contrib.auth.models import User
from demo import settings
from django.core.mail import send_mail
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.cache.backends.base import DEFAULT_TIMEOUT
from django.urls import reverse
from django.db import connection
from testapp.forms import EmployeeForm,SignUpForm,DijkstraForm
from testapp.models import Employee, MonthlyWeatherByCity, DijkstraModel

from cruds_adminlte.crud import CRUDView
from cruds_adminlte.inline_crud import InlineAjaxCRUD

from .models import Autor, Employee, MonthlyWeatherByCity, DijkstraModel

from django.views.generic.base import TemplateView
from django import forms
from cruds_adminlte.filter import FormFilter

from .forms import EmployeeForm, DataForm
from chartit import DataPool, Chart

from math import pi
#from bokeh.io import output_file, show
from bokeh.palettes import Category20c
#from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.resources import CDN

from django.http import HttpResponseBadRequest, HttpResponse, HttpResponseRedirect
from testapp.functions import handle_uploaded_file
from django.template import loader, RequestContext
from django.template.loader import render_to_string
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib as pl
pl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import io
from io import IOBase, StringIO, BytesIO
import pandas as pd

# for AP Supplier
from pandas import DataFrame, read_csv

# R
import rpy2.robjects as robjects
#import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#Creating charts and output them as images to the browser
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pylab
#from pylab import *
import PIL, PIL.Image
import base64

#Pyodbc
import pyodbc

#Basemap & geopandas purposes
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import geopandas as gpd
import shapefile
from descartes import PolygonPatch
import gmplot
import cv2
import random

import networkx as nx
import math






#CACHE_TTL = getattr(settings, 'CACHE_TTL', DEFAULT_TIMEOUT)

@login_required
def home(request):
        return render(request,'home.html')

class IndexView(TemplateView):
	template_name = 'index.html'

@login_required(login_url='http://192.168.75.145/login/')
def emp(request):
	model = Employee
	form_class = EmployeeForm
	template_name_base = 'analysis'  # customer cruds => ccruds
	namespace = None
	check_login = True
	check_perms = True
	views_available = ['edit', 'show', 'indeks']
	add_form = EmployeeForm

	# if request is not post, initialize an empty form
	form = form_class(request.POST or None)
	if request.method == "POST":
		form = EmployeeForm(request.POST)
		if form.is_valid():
			try:
				form.save()
				return redirect('/testapp/show/')
			except:
				pass
		else:
			form = EmployeeForm()
	return render(request,'indeks.html',{'form':form})

#@login_required(login_url='http://192.168.75.145/login/')
#def dijk(request):
#	model = DijkstraModel
#	form_class = DijkstraForm
#	namespace = None
#	check_login = True
#	check_perms = True
#	add_form = DijkstraForm
#
#	# if request is not post, initialize an empty form
#	dijkform = form_class(request.POST or None)
#	if request.method == "POST":
#		dijkform = DijkstraForm(request.POST)
#		if dijkform.is_valid():
#			try:
#				dijkform.save()
#				#instance=dijkform.save(commit=False)
#				#instance.save(using='regional')
#				return redirect('/testapp/dijk/')
#			except:
#				pass
#		else:
#			dijkform = DijkstraForm()
#	return render(request,'indeks_dijkstra.html',{'dijkform':dijkform})


@login_required(login_url='http://192.168.75.145/login/')
def dijk(request):
	form = DijkstraForm(request.POST or None)
	if form.is_valid():
		# nx.Graph() couldn't generate arrow or something happened
		G = nx.DiGraph()

		kranji = request.POST.get("reg1")
		garuda = request.POST.get("reg4")

		# xcheck value
		#print(kranji)
		#print(garuda)

		G.add_edges_from([("KT","KR")],weight=8.9)
		G.add_edges_from([("KT","GR")],weight=16.9)
		G.add_edges_from([("KT","GC")],weight=25.7)
		G.add_edges_from([("KT","MBP")],weight=20.1)
		#G.add_edges_from([("KR","GR")],weight=15.3)
		G.add_edges_from([("KR","GR")],weight=int(kranji))
		G.add_edges_from([("KR","KT")],weight=17.2)
		G.add_edges_from([("KT","GC")],weight=100)
		G.add_edges_from([("GR","MBP")],weight=6.6)
		G.add_edges_from([("GR","KT")],weight=9.6)
		#G.add_edges_from([("MBP","GC")], weight=6.1)
		G.add_edges_from([("MBP","GC")], weight=int(garuda))
		G.add_edges_from([("MBP","KT")],weight=16.5)

		A = nx.dijkstra_path(G, 'KR','GC')
		noCor = ["blue" if n in A else "red" for n in G.nodes()]
		
		pos = nx.spring_layout(G)
		nx.draw_networkx_nodes(G, pos=pos, node_color=noCor)
		nx.draw_networkx_labels(G, pos=pos,font_color='white')
		
		nx.draw_networkx_edges(G, pos=pos, arrows=True, with_labels=True, arrowstyle='->')
		labels = nx.get_edge_attributes(G,'weight')
		nx.draw_networkx_edge_labels(G, pos=pos ,edge_labels=labels)
		
		plt.axis('off')
		plt.savefig('/home/bigh/django-cruds-adminlte/build/lib/cruds_adminlte/static/network.png')
		# Test print result
		print(nx.dijkstra_path(G,'KR','GC'))

		
		# Truncate first
		cursor = connection.cursor()
		cursor.execute("TRUNCATE TABLE `regional2`")
		form.save()
		return HttpResponseRedirect("http://192.168.75.145/testapp/dij/")

	context = {'form': form}

	return render(request,'indeks_dijkstra.html',context)

	

def show(request):
	model = Employee
	namespace = 'testapp'
	check_login = True
	check_perm = True
	views_available = ['show','indeks','edit']
	employees = Employee.objects.all()
	page = request.GET.get('page',1)

	paginator = Paginator(employees,5)
	try:
		users = paginator.page(page)
	except PageNotAnInteger:
		users = paginator.page(1)
	except EmptyPage:
		users = paginator.page(paginator.num_pages)
	return render(request, 'show.html',{'users':users})

def edit(request,id):
	employee = Employee.objects.get(id=id)
	return render(request,'edit.html',{'employee':employee})

def update(request,id):
	employee = Employee.objects.get(id=id)
	form = EmployeeForm(request.POST,instance = employee)
	if form.is_valid():
		form.save()
		return redirect("/testapp/show")
	return render(request,'edit.html',{'employee':employee})

def destroy(request,id):
	employee = Employee.objects.get(id=id)
	employee.delete()
	return redirect("/testapp/show")

def R(request):
	return render(request,'rchar.html')

@login_required(login_url='http://192.168.75.145/login/')
def getAjax(request):
	# Bekasi - Bogor - Depok
	latitude_list = [-6.2349,-6.59444,-6.4]
	longitude_list = [106.9896,106.78917,106.81861]
	gmap1 = gmplot.GoogleMapPlotter(-6.2350,106.78916,13)
	gmap1.scatter( latitude_list, longitude_list, '#FF0000', size = 40, marker = False )
	gmap1.polygon(latitude_list, longitude_list, 'cornflowerblue', edge_width = 2.5) 
	gmap1.draw("/home/bigh/django-cruds-adminlte/demo/testapp/templates/gmplot.html")

	return render(request,'Ajax.html')
	#return render(request,'gmplot.html')

@login_required(login_url='http://192.168.75.145/login/')
def panda(request):
	return render(request,'panda.html')

def weather_chart_view(request):
	#Step 1: Create a DataPool with the data we want to retrieve
	weatherdata = DataPool(
		series = 
		[{ 'options': {
			'source': MonthlyWeatherByCity.objects.all()},
			'terms': [ 
				'month',
				'houston_temp',
				'boston_temp'
			]
		}
			
		])

	#Step 2: Create the Chart object
	cht = Chart(
		datasource = weatherdata,
		series_options =
		[{'options': {
			'type': 'line',
			'stacking': False},
		'terms': {
			'month': [
				'boston_temp',
				'houston_temp'
			]
		}
		}
		],

		chart_options =
		{'title': {
			'text': 'Weather Data of Boston and Houston'},
		'xAxis': {
			'title': {
				'text': 'Month Number'}
		}
		}
	)
	
	#Step 3: Send the chart object to the template
	return render_to_response('weatherchart.html',{'weatherchart': cht})


@login_required(login_url='http://192.168.75.145/login/')
#@cache_page(CACHE_TTL)
def visual(request):
	if request.method == 'POST':
		dataform = DataForm(request.POST,request.FILES)
		if dataform.is_valid():
			handle_uploaded_file(request.FILES['file'])
			return HttpResponse('File Uploaded Successfully')
	else:
		dataform = DataForm()
	
	#df = pd.read_html('www.fdic.gov/bank/individual/failed/banklist.html')
	# Adjust encoding type
	#dfs = pd.read_csv('/home/bigh/banklist.csv',encoding='windows-1252')
	#dfs = pd.read_csv('/home/bigh/tweets.csv',encoding='utf8')
	dfs = pd.read_excel('/home/bigh/tweets_hary.xlsx',sheet_name='tweets')
	#dfs = pd.read_excel('/home/bigh/tweets.xlsx',sheet_name='tweets')
	#df = dfs['username'].nunique()
	#df = dfs['username'].unique()
	#df = dfs['username'].value_counts()

	#topretweets = dfs.groupby('username')[['retweets']].sum()
	#df = topretweets.sort_values('retweets',ascending=False)[:10]

	#We would like to know the number of words in each tweet for 5 rows x 6 columns
	dfs['len'] = dfs['tweet '].apply(len)
	grams = dfs.head()

	#See the description of the column we just created
	#dfs['len'] = dfs['tweet '].apply(len)
	#df = dfs.describe()

	#See that the longest tweet is 158
	#dfs['len'] = dfs['tweet '].apply(len)
	#df = dfs[dfs['len']==158]

	#We can see the full tweet  with 158 long characters
	#dfs['len'] = dfs['tweet '].apply(len)
	#df = dfs[dfs['len']==158].iloc[0]

	#Merging dataframes
	#by_tweets = dfs.groupby('username')['tweet '].count().reset_index()
	#df = by_tweets

	#by_retweets = dfs.groupby('username')['retweets'].sum().reset_index()
	#df = by_retweets

	#merged_dfs = pd.merge(by_tweets, by_retweets, how='right', left_on='username', right_on='username')
	#df = merged_dfs.head()

	# Histogram, Scatter plot
	#dfs['len'] = dfs[dfs.plot(kind='hist',bins=50,figsize=(12,6))].apply(len)

	# Let's look at the most used words
	#corpus = ' '.join(dfs['tweet '])
	#corpus = corpus.replace('.', '. ')
	#wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=2400, height=2000).generate(corpus)
	#plt.figure(figsize=(12,15))
	#plt.imshow(wordcloud)
	#plt.show()

	# Store Image in a string buffer
	#buffer = BytesIO()
	#canvas = pylab.get_current_fig_manager().canvas
	#canvas.draw()
	#pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
	#pilImage.save(buffer, "PNG")
	#pylab.close()

	#df = buffer.getvalue()

	# I would like to see what MESTAfrica tweeted about in 2017
	#mest = dfs[dfs['username']=='MESTAfrica']
	mest = dfs[dfs['username']=='tifosilinux']
	corpu = ' '.join(dfs['tweet '])
	corpu = corpu.replace('.','. ')
	wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=2400, height=2000).generate(corpu)
	plt.figure(figsize=(12,15))
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()

	buffer = BytesIO()
	canvas = pylab.get_current_fig_manager().canvas
	canvas.draw()
	pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
	pilImage.save(buffer, "PNG")
	pylab.close()

	#df = buffer.getvalue()
	df = base64.b64encode(buffer.getvalue()).decode('ascii')

	#########################################################################
	x = [1,3,5,7,9,11,13]
	y = [1,2,3,4,5,6,7]
	title = 'y = f(x)'
	
	plot = figure(title = title,
		x_axis_label = 'X-Axis',
		y_axis_label = 'Y-Axis',
		plot_width = 400,
		plot_height = 400)

	plot.line(x, y, legend= 'f(x)', line_width = 2)
	#Store Components
	script, div = components(plot)

	#Matplotlib Purposes
	COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_week', 'native_country', 'label']
	#PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
	PATH = "/home/bigh/adult.data"
	df_train = pd.read_csv(PATH,
			skipinitialspace=True,
			names = COLUMNS,
			index_col=False)
	ES1 = df_train.groupby(['label', 'marital'])['capital_gain'].mean().unstack()
	ES = ES1.plot.bar()
	fig = ES.get_figure()
	fig.savefig("/home/bigh/django-cruds-adminlte/demo/testapp/static/adult.png")

	#Feed them to the django template.
	#return render(request,'visual.html',{'form':dataform,'htmllink':df,'script': script, 'div':div},content_type="image/png")
	#Somehow logout/ change passwd on navbar-custom menu didn't working with this
	#return render_to_response('visual.html',{'form':dataform,'script':script,'div':div,'htmllink': df,'grams':grams})
	return render(request,'visual.html',{'form':dataform,'script':script,'div':div,'htmllink': df,'grams':grams})
	#return HttpResponse(df, content_type="image/png")

def simple_chart(request):
	plot = figure()
	plot.circle([1,2],[3,4])
	script, div = components(plot, CDN)
	return render(request, "simple_chart.html", {"the_script": script, "the_div": div})


def getNums(request):
	n = np.array([2,3,4])
	name1 = "name-" + str(n[1])
	return HttpResponse('{"name":"'+name1+'","age":29,"city":"New York"}')

def getAvg(request):
	s1 = request.GET.get("val","")
	if len(s1)==0:
		return HttpResponse("none")
	l1=s1.split(',')
	ar=np.array(l1,dtype=int)
	
	return HttpResponse(str(np.average(ar)))

def getImages(request):

	x = np.arange(0,2 * np.pi, 0.01)
	s = np.cos(x) ** 2
	plt.plot(x,s)
	
	plt.xlabel('xlabel(X)')
	plt.ylabel('ylabel(Y)')
	plt.title('Simple graph!')
	plt.grid(True)

	response = HttpResponse(content_type="image/jpeg")
	plt.savefig(response,format="png")
	return response

@login_required(login_url='http://192.168.75.145/login/')
def getDatas(request):
	
	#fields=['AP_No','AP_SupplierName','AP_PayAmount']
	#df = pd.read_csv('/home/bigh/AP_History2.csv',error_bad_lines=False,sep=';',chunksize=10000000, index_col=['AP_No'], usecols=fields)
	#x = df['AP_SupplierName']
	#y = df['AP_PayAmount']

	#z = plt.plot(x,y)
	
	#fig = z.get_figure()
	#fig.savefig('/home/bigh/django-cruds-adminlte/demo/testapp/static/ap.png')

	#return HttpResponse(df)

	COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_week', 'native_country', 'label']
	PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
	df_train = pd.read_csv(PATH,
        	skipinitialspace=True,
        	names = COLUMNS,
        	index_col=False)
	#return HttpResponse(df_train.shape)
	#df_train.groupby(['label'])['age'].min()
	#ES = df_train.groupby(['label', 'marital'])['capital_gain'].max()
	ES1 = df_train.groupby(['label', 'marital'])['capital_gain'].mean().unstack()
	ES = ES1.plot.bar()
	fig = ES.get_figure()
	fig.savefig('/home/bigh/django-cruds-adminlte/demo/testapp/static/adult.png')
	#return HttpResponse(df_train.groupby(['label'])['age'].min())
	return render(request,'S.html',{'S':ES})
	#return HttpResponse(df_train.groupby(['label', 'marital'])['capital_gain'].max())
	

@login_required(login_url='http://192.168.75.145/login/')
def getR(request):
	#v = robjects.FloatVector([1.1,2.2,3.3,4.4,5.5,6.6])
	#m = robjects.r['matrix'](v, nrow=2)
	
	#letters = robjects.r['letters']
	#rcode = 'paste(%s, collapse="-")' %(letters.r_repr())
	#res = robjects.r(rcode)

	#robjects.r('x=c()')
	#robjects.r('x[1]=22')
	#robjects.r('x[2]=44')
	#objek = robjects.r('x')

	# Looking for Mean
	v1 = robjects.FloatVector([9,5,7,12,0,9,34,5,12,10,-4])
	m1 = robjects.r['mean'](v1)
	# Looking for Median
	v2 = robjects.FloatVector([12,7,3,4.2,18,2,54,-21,8,-5])
	m2 = robjects.r['median'](v2)
	# Looking for Linier Regresion
	#x = robjects.FloatVector(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
	#y = robjects.FloatVector(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)
	#m3 = robjects.r['lm'](y~x)

	pis = robjects.r['pi']

	fields=['model','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
	df = pd.DataFrame(pd.read_csv("/home/bigh/mtcars.csv",index_col=['model']))
	r_dataframe = pandas2ri.py2ri(df)

	return render(request,'R.html',{'R5':m1,'R6':m2, 'R4':r_dataframe,'R3':pis})

@login_required(login_url='http://192.168.75.145/login/')
def getData(request):
	dfs = pd.read_excel('/home/bigh/HPO.xlsx',sheet_name='hpo')
	dff = dfs['HPO_SupplierAddress2'].value_counts()
	
	plt.figure(figsize=(10,13))
	plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.18)
	df = dff.plot.bar(alpha=0.5,title='AP History')
	df.set_xlabel("Supplier Address")
	df.set_ylabel("Frequency")
	
	buffer = BytesIO()
	canvas = pylab.get_current_fig_manager().canvas
	canvas.draw()
	pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
	pilImage.save(buffer, "PNG")
	pylab.close()
	
	df = base64.b64encode(buffer.getvalue()).decode('ascii')


	#Pyodbc
	#cnxn = pyodbc.connect("DRIVER={MySQL};SERVER=localhost;DATABASE=demo;UID=root;PASSWORD=johndoe;")
	#ptdBase = cnxn.cursor()
	#MySQLQuery = """
	#	SELECT * from testapp_customer
	#"""
	#rows = ptdBase.execute(MySQLQuery)
	#for rwx in rows:
	#	rwx


	#Basemap & Shapefiles
	#sf = gpd.read_file("/home/bigh/basemap/INDONESIA_PROP.shp") # Provinsi
	sf = gpd.read_file("/home/bigh/basemap/kabupaten/INDONESIA_KAB.shp") # Kabupaten
	sg = gpd.read_file("/home/bigh/basemap/kota/IND_KOT_point.shp") # Kota
	#sf.plot(column='Provinsi',categorical=True,figsize=(14,8)) # Plot Provinsi
	#sf.plot(column='KECAMATAN',categorical=True,legend=True,cmap='tab20',figsize=(40,20)) # Plot Kabupaten
	#sf1 = sf[(sf.KECAMATAN == "AIR BATU")]
	#sf1.plot(column='KECAMATAN',categorical=True,legend=True,cmap='tab20')
	# OR
	sf1 = sf[sf.Kabupaten_.isin(['BOGOR','BANDUNG','BEKASI','BONDOWOSO','BANTUL','SLEMAN'])]
	sf1.plot(column='Kabupaten_',categorical=True,legend=True,cmap='tab20',figsize=(14,8)) # This should not be declare repeatedly
	# Get All attribute information
	#h = gpd.GeoDataFrame(sf).iloc[1]
	#sf = gpd.read_file("/home/bigh/basemap/allfrombps/podes_bps2014.shp") # All Indonesia
	#h = gpd.GeoDataFrame(sf).iloc[1]
	#sf.plot(figsize=(12,6))
	#sf.plot(categorical=True,figsize=(8,4))
	
	#sf=shapefile.Reader('/home/bigh/basemap/allfrombps/podes_bps2014.shp')
	#poly=sf.shape(1).__geo_interface__
	#fig = plt.figure() 
	#axs = fig.gca() 
	#axs.add_patch(PolygonPatch(poly, fc='#ffffff', ec='#000000', alpha=0.5, zorder=2 ))
	#axs.axis('scaled')
	#plt.show()

	buffered = BytesIO()
	canvases = pylab.get_current_fig_manager().canvas
	canvases.draw()
	pilImages = PIL.Image.frombytes("RGB", canvases.get_width_height(), canvases.tostring_rgb())
	pilImages.save(buffered, "PNG")
	pylab.close()
	m = base64.b64encode(buffered.getvalue()).decode('ascii')

	sg1 = sg[sg.Nama.isin(['YOGYAKARTA','TANGGERANG','JAKARTA','Depok','BEKASI','BANDUNG'])]
	sg1.plot(column='Nama',categorical=True,legend=True,cmap='tab20',figsize=(8,4)) # This should not be declare repeatedly
	buffered1 = BytesIO()
	canvases1 = pylab.get_current_fig_manager().canvas
	canvases1.draw()
	pilImages1 = PIL.Image.frombytes("RGB", canvases1.get_width_height(), canvases1.tostring_rgb())
	pilImages1.save(buffered1, "PNG")
	pylab.close()
	n = base64.b64encode(buffered1.getvalue()).decode('ascii')
	
	#Others
	#ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
	ts = pd.Series(np.random.randn(1000), index=pd.date_range(start='1/12/2018',end='3/4/2019',periods=1000))
	ts = ts.cumsum()
	ax = ts.plot()
	fig = ax.get_figure()
	#return HttpResponse(ts.plot())

	df2 = pd.DataFrame(np.random.rand(10, 4), columns=['Linux', 'Windows', 'Mac', 'Solaris'])
	ax2 = df2.plot.bar(figsize=(4.5,4.5))
	figs = ax2.get_figure()
	figs.savefig('/home/bigh/django-cruds-adminlte/demo/testapp/static/plot.png')


	series = pd.Series(3 * np.random.rand(4), index=['(L)','(W)','(M)','(S)'],name='Operating System Dist.')
	ax3 = series.plot.pie(figsize=(4.5,4.5))
	figs2 = ax3.get_figure()
	figs2.savefig('/home/bigh/django-cruds-adminlte/demo/testapp/static/pie.png')

	df3 = pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=list('ABCD'))
	df3 = df3.cumsum()
	ax3 = df3.plot(legend=False,figsize=(9,4.5))
	figs3 = ax3.get_figure()
	figs3.savefig('/home/bigh/django-cruds-adminlte/demo/testapp/static/nolegend.png')

	template = loader.get_template('pandas.html')
	context = {}


	### Analytic from OpenCV
	cap = cv2.VideoCapture('/home/bigh/django-cruds-adminlte/demo/testapp/static/upload/sources/cars.mp4')
	ret, frame = cap.read()
	ratio = .5 # resize ratio
	image = cv2.resize(frame, (0,0), None, ratio, ratio) # resize image
	
	df5 = pd.read_csv('/home/bigh/django-cruds-adminlte/demo/testapp/static/upload/later_analysis.csv') # reads csv file and makes it a dataframe
	rows, columns = df5.shape # shape of dataframe
	#print('Rows:',rows)
	#print('Columns:',columns)

	fig5 = plt.figure(figsize=(10,8)) # width and height of image
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # plots first frame of video

	for i in range(columns - 1): # loops through all columns of dataframes, -1 since index is counted
		y = df5.loc[df5[str(i)].notnull(), str(i)].tolist() # grabs not null data from columns
		df7 = pd.DataFrame(y, columns=['xy']) # create another dataframe with only one column
		
		# create another dataframe where it splits centroids x and y values into two columns
		df8 = pd.DataFrame(df7['xy'].str[1:-1].str.split(',',expand=True).astype(float))
		df8.columns = ['x','y'] # renames columns

		# plots series with random colors
		plt.plot(df8.x, df8.y, marker='x', color=[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)], label='ID: ' + str(i))

	# plot info
	plt.title('Tracking of centroids')
	plt.xlabel('X Position')
	plt.ylabel('Y Position')
	plt.legend(bbox_to_anchor=(1,1.2), fontsize='x-small') # legend location and font
	#plt.show()
	plt.close()
	fig5.savefig('/home/bigh/django-cruds-adminlte/build/lib/cruds_adminlte/static/traf.png') # saves image


	#return HttpResponse(template.render(context, request))
	#Somehow logout/ change passwd on navbar-custom menu didn't working with this
	#return render_to_response('pandas.html',{'supplier':df,'bm':m,'bn':n})
	return render(request,'pandas.html',{'supplier':df,'bm':m,'bn':n})

def getImage(request):
    import django
    import datetime

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

def signup(request):
	if request.method == 'POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			user = form.save(commit=False)
			user.is_active = False
			user.save()

			current_site = get_current_site(request)
			subject = "Activate Your Tifosilinux Account"
			message = render_to_string('account_activation_email.html',{
				'user': user,
				'domain': current_site.domain,
				'uid': urlsafe_base64_encode(force_bytes(user.pk)).decode(),
				'token': account_activation_token.make_token(user),
			})
			user.email_user(subject, message)

			to = form.cleaned_data.get('email')
			send_mail(subject,message, settings.EMAIL_HOST_USER,[to])
			
			return redirect('account_activation_sent')

	else:
		form = SignUpForm()
	return render(request,'signup.html',{'form':form})

def account_activation_sent(request):
        return render(request,'account_activation_sent.html')

def activate(request, uidb64, token):
        try:
                uid = force_text(urlsafe_base64_decode(uidb64))
                user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                user = None

        if user is not None and account_activation_token.check_token(user, token):
                user.is_active = True
                user.profile.email_confirmed = True
                user.save()
                login(request, user)
                return redirect('home')
        else:
                return render(request, 'account_activation_invalid.html')
