from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import get_object_or_404, render , redirect
from .models import Specimen
from django.template import loader
from django.views import generic
from .forms import *
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
import pandas as pd
from django.contrib import messages



#constants

columns_names = [
"code",
"label",
"notes",
"old_code",
"collection_day",
"collection_month",
"collection_year",
"death_date",
"sex_code",
"refrigerator",
"tray",
"row" ,
"column",
"location_code",
"location",
"genus",
"species",
"country",
"province",
"latitude",
"longitude",
"elevation",
"light_dark",
"histology_location" ,
"histology_stage_performed",
"histology_stage_next_up",
"rna_location" ,
"etoh_voucher" ,
"elytron",
"purpose"
]
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
              try:
                  especimenes_list = especimenes_list.filter(code__exact= form.cleaned_data["code"])
              except:
                  print("Invalid code")
          if form.cleaned_data["label"]:
              especimenes_list = especimenes_list.filter(label__icontains= form.cleaned_data["label"] )
          if form.cleaned_data["notes"] :
              especimenes_list = especimenes_list.filter(notes__icontains= form.cleaned_data["notes"] )
          if form.cleaned_data["old_code"]:
              especimenes_list = especimenes_list.filter(old_code__icontains= form.cleaned_data["old_code"] )
          if form.cleaned_data["collection_day"]:
              especimenes_list = especimenes_list.filter(collection_day__icontains= form.cleaned_data["collection_day"] )
          if form.cleaned_data["collection_year"]:
              try:
                  especimenes_list = especimenes_list.filter(collection_year__exact= form.cleaned_data["collection_year"] )
              except:
                  print("Invalid code")

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

def handle_uploaded_file(f, request):
    print("°°°°°°°°°°°file is being handled°°°°°°°°°°°°°°°°°°°°")
    with open("temp.txt", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    df = pd.read_csv("temp.txt", sep="\t", header=0, names = columns_names)
    total = 0
    #print("°°°°°°°°°°°Dataframe°°°°°°°°°°°°°°°°°°°°")
    #print(df)
    for index, row in df.iterrows():
        dict = {}
        for name in columns_names:
            print("°°°°°°°°°°° print name and row name°°°°°°°°°°°°°°°°°°°°")
            print(name)
            print(row[name])
            dict[name] = row[name]
        print("dict")
        print(dict)

        if Specimen.objects.get(code = dict["code"]):
            print("Already there")
            print(dict["code"])
            pass
        else:
            print("creating new with info ")
            print(dict)
            new_specimen = Specimen.objects.create(**dict)
            new_specimen.save()
            total = total+1
    if total ==0:
        messages.add_message(request, messages.INFO, f"Attention! No specimens were added")
    else:
        messages.add_message(request, messages.INFO, f"Success! {total} specimen(s) were added")
@login_required
def upload(request):
    template = loader.get_template("inventory/bootstrap/upload.html")
    context = {}
    if request.method == "POST":
        print("°°°°°°°°°°°Method is post°°°°°°°°°°°°°°°°°°°°")
        form= UploadFileForm(request.POST, request.FILES)


        if form.is_valid():
            print("°°°°°°°°°°°form is valid°°°°°°°°°°°°°°°°°°°°")

            handle_uploaded_file(request.FILES["file"],request)
    else:
        form = UploadFileForm()


    return render(request,"inventory/bootstrap/upload.html", {"form":form} )

import csv

@login_required
def download(request):
    template = loader.get_template("inventory/bootstrap/download.html")
    specimens = Specimen.objects.all()

    response = HttpResponse(
        content_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="catalog.txt"'},
    )

    writer = csv.writer(response, delimiter="\t")

    row_data_header = []
    for name in columns_names:
        row_data_header.append(name)
    writer.writerow(row_data_header)

    for specimen in specimens:
        row_data = [specimen.code, specimen.label, specimen.notes, specimen.old_code, specimen.collection_day, specimen.collection_month, specimen.collection_year , specimen.death_date, specimen.sex_code, specimen.refrigerator, specimen.tray, specimen.row, specimen.column, specimen.location_code, specimen.location, specimen.genus, specimen.species, specimen.country, specimen.province, specimen.latitude, specimen.longitude, specimen.elevation, specimen.light_dark, specimen.histology_location, specimen.histology_stage_performed, specimen.histology_stage_next_up, specimen.rna_location, specimen.etoh_voucher, specimen.elytron, specimen.purpose.rstrip()]
        writer.writerow(row_data)


    return response


def specimen(request, specimen_code):
    especimenes_list = Specimen.objects.filter(code__exact = specimen_code)
    specimen = get_object_or_404(Specimen, pk=specimen_code)
    template = loader.get_template("inventory/bootstrap/specimen.html")

    if specimen:
        try:

            context = {
                "specimen": specimen,
            }
            return HttpResponse(template.render(context, request))
        except (KeyError, Specimen.DoesNotExist):
            # Redisplay the question voting form.
            context = {
                "error_message": "Specimen does not exist",
            }
            return HttpResponse(template.render(context, request))

def login_view(request):
    template_path = "inventory/bootstrap/login.html"
    template = loader.get_template(template_path)
    context = {
    }
    if request.method == "POST" :
        form = LoginForm(request.POST)
        #dir(request.POST)
        print("Login succesful")
        username = request.POST["username"]
        password = request.POST["password"]
        next = ""
        try:
            next = request.GET["next"]
        except:
            pass
        user = authenticate(request, username=username, password=password)
        if user is not None:

            login(request, user)
            print("Next value: ")
            print(next)
            if next == "":
                return redirect("/")
            return redirect(next)

        return render(request, template_path, {"form":form} )
    else:
        form = LoginForm()
        return render(request, template_path, {"form":form} )

def logout_view(request):
    logout(request)
    return redirect("/")
