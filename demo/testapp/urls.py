from cruds_adminlte.urls import crud_for_app
from testapp.forms import EmployeeForm

app_name = 'testapp'

urlpatterns = []


urlpatterns += crud_for_app(app_name, check_perms=True, namespace="ns")
urlpatterns += crud_for_app(app_name,login_required=True,check_perms=True,cruds_url='emp')
urlpatterns += crud_for_app(app_name,login_required=True,check_perms=True,cruds_url='show')
urlpatterns += crud_for_app(app_name,login_required=True,check_perms=True,cruds_url='edit')
