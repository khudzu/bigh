"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls import url, include
from django.contrib import admin

from cruds_adminlte.urls import crud_for_app
from testapp.views import (IndexView, emp, simple_chart, getNums,getAvg, getImage)
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from testapp import views,views_logic, views_detection, views_dijkstra, views_gmplot
from django.urls import path

from django.contrib.auth.models import User
from rest_framework import serializers, viewsets, routers
from django.views.decorators.cache import cache_page



# Serializers define the API representation.
class UserSerializer(serializers.HyperlinkedModelSerializer):
	class Meta:
		model = User
		fields = ('url', 'username', 'email', 'is_staff')


# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
	queryset = User.objects.all()
	serializer_class = UserSerializer


# Routers provide a way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'users', UserViewSet)

urlpatterns = [
    url(r'^$', IndexView.as_view()),

    url(r'^accounts/login/$', auth_views.login, name='login'),
    #url(r'^logout/$', auth_views.logout, name='logout'), # It is being used by custom login-logout
    url(r'^admin/', admin.site.urls),
    url(r'^select2/', include('django_select2.urls')),
    url('^namespace/', include('testapp.urls')),
	
	url(r'testapp/emp/',views.emp, name='testapp_emp_list'),
	url(r'testapp/dijk/',views_dijkstra.edged, name='testapp_dijkstra'),
	url(r'testapp/dij/',views.dijk, name='testapp_dijkstra'),
	url(r'testapp/rchar/',views.R, name='testapp_workspace'),
	url(r'testapp/weatherchart/',views.weather_chart_view, name='testapp_workspace2'),
	#url(r'testapp/visual/',cache_page(60*60)(views.visual), name='testapp_bokeh'),
	#Visual below with cache redis
	url(r'testapp/visual/',views.visual, name='testapp_bokeh'),
	url(r'testapp/simple_chart',views.simple_chart, name='simple_chart'),
	
	url(r'^$',views.home, name='home'),
        url(r'^login/$',auth_views.login,{'template_name': 'login.html'},name='login'),
        url(r'^logout/$',auth_views.logout,{'next_page': 'login'},name='logout'),
        url(r'^signup/$',views.signup,name='signup'),
	url(r'^account_activation_sent/$',views.account_activation_sent, name='account_activation_sent'),
	url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$', views.activate, name='activate'),
	url(r'testapp/panda/',views.panda, name='testapp_workspace1'),
	url(r'^testapp/pandas/',views.getData, name='testapp_pandas'),
	url(r'^testapp/getnum/',views.getNums),
	url(r'^testapp/getavg/',views.getAvg),
	url(r'^testapp/getimg/',views.getImage),
	url(r'^testapp/R/',views.getR, name='testapp_R'),
	url(r'^testapp/ajx/',views.getAjax, name='testapp_ajax'),
	url(r'^testapp/gmplot/',views_gmplot.getgmPlot),
	url(r'^testapp/getdata/',views.getDatas),
        url(r'testapp/show/',views.show),
	path('testapp/edit/<int:id>',views.edit),
	path(r'testapp/update/<int:id>',views.update),
	path(r'testapp/destroy/<int:id>',views.destroy),
	url(r'^',include(router.urls)),
	url(r'^api-auth/',include('rest_framework.urls',namespace='rest_framework')),
	url(r'^testapp/detect/',views_detection.getDetect, name='testapp_motion'),
	url(r'^testapp/for/',views_logic.forStar),
]


#urlpatterns += crud_for_app('auth', login_required=True, cruds_url='lte')
urlpatterns += crud_for_app('auth',login_required=True,cruds_url='lte')


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
