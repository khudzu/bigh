def handle_uploaded_file(f):
	with open('testapp/static/upload/sources/'+f.name,'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)
