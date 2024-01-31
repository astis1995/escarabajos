from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from .models import Specimen
from django.template import loader
from django.views import generic
from .forms import SpecimenSearchForm
#import HttpRequest

def index(request):
    template = loader.get_template("inventory/bootstrap/tables.html")
    especimenes_list = Specimen.objects.order_by("code")[:]
    context = {
        "especimenes_list": especimenes_list,
    }

    if request.method == "POST":
        form = SpecimenSearchForm(request.POST)
        field_names = list(form.fields.keys())
        print(field_names)
        if form.is_valid():

            #code = form.cleaned_data["code"]
            #genus = form.cleaned_data["genus"]
            #species = form.cleaned_data["species"]
            #collection_year = form.cleaned_data["collection_year"]
            #especimenes_list = Specimen.objects.all()

            #if :
            #    especimenes_list = especimenes_list.filter(__icontains=)
          if form.cleaned_data["code"]:
              especimenes_list = especimenes_list.filter(code__exact= form.cleaned_data["code"])
          if form.cleaned_data["label"]:
              especimenes_list = especimenes_list.filter(label__icontains= form.cleaned_data["label"] )
          if form.cleaned_data["notes"] :
              especimenes_list = especimenes_list.filter(notes__icontains= form.cleaned_data["notes"] )
          if form.cleaned_data["old_code"]:
              especimenes_list = especimenes_list.filter(old_code__icontains= form.cleaned_data["old_code"] )
          if form.cleaned_data["collection_day"]:
              especimenes_list = especimenes_list.filter(collection_day__icontains= form.cleaned_data["collection_day"] )
          if form.cleaned_data["collection_year"]:
              especimenes_list = especimenes_list.filter(collection_year__icontains= form.cleaned_data["collection_year"] )
          if form.cleaned_data["death_date"]:
              especimenes_list = especimenes_list.filter(death_date__icontains= form.cleaned_data["death_date"] )
          if form.cleaned_data["sex_code"]:
              especimenes_list = especimenes_list.filter(sex_code__icontains= form.cleaned_data["sex_code"] )
          if form.cleaned_data["refrigerator"]:
              especimenes_list = especimenes_list.filter(refrigerator__icontains=form.cleaned_data["refrigerator"])
          if form.cleaned_data["tray"]:
              especimenes_list = especimenes_list.filter(tray__icontains= form.cleaned_data["tray"])
          if form.cleaned_data["row"]:
              especimenes_list = especimenes_list.filter(row__icontains= form.cleaned_data["row"] )
          if form.cleaned_data["location"]:
              especimenes_list = especimenes_list.filter(location__icontains= form.cleaned_data["location"])
          if form.cleaned_data["location_code"]:
              especimenes_list = especimenes_list.filter(location_code__icontains= form.cleaned_data["location_code"] )

          if form.cleaned_data["column"]:
              especimenes_list = especimenes_list.filter(column__icontains= form.cleaned_data["column"])
          if form.cleaned_data["genus"]:
              especimenes_list = especimenes_list.filter(genus__icontains= form.cleaned_data["genus"])
          if form.cleaned_data["species"]:
              especimenes_list = especimenes_list.filter(species__icontains= form.cleaned_data["species"])

          if form.cleaned_data["country"]:
              especimenes_list = especimenes_list.filter(country__icontains= form.cleaned_data["country"])
          if form.cleaned_data["province"]:
              especimenes_list = especimenes_list.filter(province__icontains= form.cleaned_data["province"])
          if form.cleaned_data["latitude"]:
              especimenes_list = especimenes_list.filter(latitude__icontains= form.cleaned_data["latitude"])
          if form.cleaned_data["longitude"]:
              especimenes_list = especimenes_list.filter(longitude__icontains= form.cleaned_data["longitude"])
          if form.cleaned_data["elevation"]:
              especimenes_list = especimenes_list.filter(elevation__icontains= form.cleaned_data["elevation"])
          if form.cleaned_data["light_dark"]:
              especimenes_list = especimenes_list.filter(light_dark__icontains= form.cleaned_data["light_dark"])
          if form.cleaned_data["histology_location"]:
              especimenes_list = especimenes_list.filter(histology_location__icontains= form.cleaned_data["histology_location"])
          if form.cleaned_data["histology_stage_next_up"]:
              especimenes_list = especimenes_list.filter(histology_stage_next_up__icontains= form.cleaned_data["histology_stage_next_up"])
          if form.cleaned_data["histology_stage_performed"]:
              especimenes_list = especimenes_list.filter(histology_stage_performed__icontains= form.cleaned_data["histology_stage_performed"])
          if form.cleaned_data["purpose"]:
              especimenes_list = especimenes_list.filter(purpose__icontains= form.cleaned_data["purpose"])
          context = {
                "especimenes_list": especimenes_list,
                "field_names": field_names
            }
          template = loader.get_template("inventory/bootstrap/tables.html")

          return HttpResponse(template.render(context, request))

        print(request.POST.get("csrfmiddlewaretoken"))
        print("aylmaaaaaaoooooPOST")


    return HttpResponse(template.render(context, request))


def results(request):
    especimenes_list = {}
    if request.method == "POST":

        code_string= SpecimenSearchForm(request.POST).cleaned_data["code"]

        if form.is_valid():
            especimenes_list = Specimen.objects.filter(code__exact= code_string)


    template = loader.get_template("inventory/bootstrap/tables.html")
    context = {
        "especimenes_list": especimenes_list,
    }
    return HttpResponse(template.render(context, request))
