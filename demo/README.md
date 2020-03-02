# bigh

##########################################################
		PRODUCED DJANGO WITH GUNICORN
##########################################################
# Initiated by hary
#
# 2019

- Change directory, where the demo moduled has been existed. [/home/bigh/django-cruds-adminlte/demo/]
- Run : gunicorn --bind 192.168.75.145:8000 demo.wsgi:application --log-level=debug --timeout=600000 --access-logfile /home/bigh/nanda.log --workers 3 --bind unix:/home/bigh/data_analysis.sock
- Access by URL http://192.168.75.145/


##########################################################

