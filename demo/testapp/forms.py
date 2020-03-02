from crispy_forms.bootstrap import TabHolder, Tab, FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Submit, HTML
from django import forms
from django.utils.translation import ugettext_lazy as _
from image_cropping import ImageCropWidget

from cruds_adminlte import (DatePickerWidget,
                            TimePickerWidget,
                            DateTimePickerWidget,
                            ColorPickerWidget,
                            CKEditorWidget)
from testapp.models import Employee, DijkstraModel
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
	birth_date = forms.DateField(help_text='Required. Format: YYYY-MM-DD')
	email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')

	class Meta:
		model = User
		fields = ('username','birth_date','email','password1','password2')

class EmployeeForm(forms.ModelForm):
	class Meta:
		model = Employee
		fields = "__all__"

class DataForm(forms.Form):
	file = forms.FileField() # for creating file input

class DijkstraForm(forms.ModelForm):
	class Meta:
		model = DijkstraModel
		fields = "__all__"
		# Or
		# fields = ["reg1","reg4"]
