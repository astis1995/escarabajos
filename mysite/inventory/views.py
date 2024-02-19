from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import get_object_or_404, render , redirect
from .models import *
from django.template import loader
from django.views import generic
from .forms import *
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
import pandas as pd
from django.contrib import messages
from decimal import *
import re
from pandas.errors import ParserError
from django.core.files.uploadedfile import UploadedFile
import os
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
    specimen_list = Specimen.objects.order_by("code")[:]
    context = {
        "specimen_list": specimen_list,
    }
    print(os.getcwd())
    if request.method == "POST":
        form = SpecimenSearchForm(request.POST)
        field_names = list(form.fields.keys())
        print(field_names)
        if form.is_valid():

            #code = form.cleaned_data["code"]
            #genus = form.cleaned_data["genus"]
            #species = form.cleaned_data["species"]
            #collection_year = form.cleaned_data["collection_year"]
            #specimen_list = Specimen.objects.all()

            #if :
            #    specimen_list = specimen_list.filter(__icontains=)

          if form.cleaned_data["code"]:
              try:
                  specimen_list = specimen_list.filter(code__icontains= form.cleaned_data["code"])
              except:
                  print("Invalid code")
          if form.cleaned_data["label"]:
              specimen_list = specimen_list.filter(label__icontains= form.cleaned_data["label"] )
          if form.cleaned_data["notes"] :
              specimen_list = specimen_list.filter(notes__icontains= form.cleaned_data["notes"] )
          if form.cleaned_data["old_code"]:
              specimen_list = specimen_list.filter(old_code__icontains= form.cleaned_data["old_code"] )
          if form.cleaned_data["collection_day"]:
              specimen_list = specimen_list.filter(collection_day__icontains= form.cleaned_data["collection_day"] )
          if form.cleaned_data["collection_year"]:
              try:
                  specimen_list = specimen_list.filter(collection_year__exact= form.cleaned_data["collection_year"] )
              except:
                  print("Invalid code")

          if form.cleaned_data["death_date"]:
              specimen_list = specimen_list.filter(death_date__icontains= form.cleaned_data["death_date"] )
          if form.cleaned_data["sex_code"]:
              specimen_list = specimen_list.filter(sex_code__icontains= form.cleaned_data["sex_code"] )
          if form.cleaned_data["refrigerator"]:
              specimen_list = specimen_list.filter(refrigerator__icontains=form.cleaned_data["refrigerator"])
          if form.cleaned_data["tray"]:
              specimen_list = specimen_list.filter(tray__icontains= form.cleaned_data["tray"])
          if form.cleaned_data["row"]:
              specimen_list = specimen_list.filter(row__icontains= form.cleaned_data["row"] )
          if form.cleaned_data["location"]:
              specimen_list = specimen_list.filter(location__icontains= form.cleaned_data["location"])
          if form.cleaned_data["location_code"]:
              specimen_list = specimen_list.filter(location_code__icontains= form.cleaned_data["location_code"] )

          if form.cleaned_data["column"]:
              specimen_list = specimen_list.filter(column__icontains= form.cleaned_data["column"])
          if form.cleaned_data["genus"]:
              specimen_list = specimen_list.filter(genus__icontains= form.cleaned_data["genus"])
          if form.cleaned_data["species"]:
              specimen_list = specimen_list.filter(species__icontains= form.cleaned_data["species"])

          if form.cleaned_data["country"]:
              specimen_list = specimen_list.filter(country__icontains= form.cleaned_data["country"])
          if form.cleaned_data["province"]:
              specimen_list = specimen_list.filter(province__icontains= form.cleaned_data["province"])
          if form.cleaned_data["latitude"]:
              specimen_list = specimen_list.filter(latitude__icontains= form.cleaned_data["latitude"])
          if form.cleaned_data["longitude"]:
              specimen_list = specimen_list.filter(longitude__icontains= form.cleaned_data["longitude"])
          if form.cleaned_data["elevation"]:
              specimen_list = specimen_list.filter(elevation__icontains= form.cleaned_data["elevation"])
          if form.cleaned_data["light_dark"]:
              specimen_list = specimen_list.filter(light_dark__icontains= form.cleaned_data["light_dark"])
          if form.cleaned_data["histology_location"]:
              specimen_list = specimen_list.filter(histology_location__icontains= form.cleaned_data["histology_location"])
          if form.cleaned_data["histology_stage_next_up"]:
              specimen_list = specimen_list.filter(histology_stage_next_up__icontains= form.cleaned_data["histology_stage_next_up"])
          if form.cleaned_data["histology_stage_performed"]:
              specimen_list = specimen_list.filter(histology_stage_performed__icontains= form.cleaned_data["histology_stage_performed"])
          if form.cleaned_data["purpose"]:
              specimen_list = specimen_list.filter(purpose__icontains= form.cleaned_data["purpose"])
          context = {
                "specimen_list": specimen_list,
                "field_names": field_names
            }
          template = loader.get_template("inventory/bootstrap/tables.html")

          return HttpResponse(template.render(context, request))

        print(request.POST.get("csrfmiddlewaretoken"))

    return HttpResponse(template.render(context, request))


def results(request):
    specimen_list = {}
    if request.method == "POST":

        code_string= SpecimenSearchForm(request.POST).cleaned_data["code"]

        if form.is_valid():
            specimen_list = Specimen.objects.filter(code__exact= code_string)


    template = loader.get_template("inventory/bootstrap/tables.html")
    context = {
        "specimen_list": specimen_list,
    }
    return HttpResponse(template.render(context, request))

def handle_uploaded_catalog_file(f, request):
    print("°°°°°°°°°°°file is being handled°°°°°°°°°°°°°°°°°°°°")
    with open("temp.txt", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    df = pd.read_csv("temp.txt", sep="\t", header=0, names = columns_names, dtype={
    "code": "string",
    "label": "string",
    "notes": "string",
    "old_code": "string",
    "collection_day": "string",
    "collection_month": "string",
    "collection_year": "string",
    "death_date": "string",
    "sex_code": "string",
    "refrigerator": "string",
    "tray": "string",
    "row": "string",
    "column": "string",
    "location_code": "string",
    "location": "string",
    "genus": "string",
    "species": "string",
    "country": "string",
    "province": "string",
    "latitude": "string",
    "longitude": "string",
    "elevation": "string",
    "light_dark": "string",
    "histology_location": "string",
    "histology_stage_performed": "string",
    "histology_stage_next_up": "string",
    "rna_location": "string",
    "etoh_voucher": "string",
    "elytron": "string",
    "purpose": "string"
    })
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

        if len(Specimen.objects.filter(code__iexact = dict["code"])) != 0 :
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
def upload_catalog(request):
    template = loader.get_template("inventory/bootstrap/upload_catalog.html")
    context = {}
    if request.method == "POST":
        print("°°°°°°°°°°°Method is post°°°°°°°°°°°°°°°°°°°°")
        form= UploadFileForm(request.POST, request.FILES)


        if form.is_valid():
            print("°°°°°°°°°°°form is valid°°°°°°°°°°°°°°°°°°°°")
            try:
                handle_uploaded_catalog_file(request.FILES["file"],request)
            except (ParserError, UnicodeDecodeError):
                messages.add_message(request, messages.INFO, f"File format is invalid. Please follow the instructions carefully or contact site administrator")
    else:
        form = UploadFileForm()


    return render(request,"inventory/bootstrap/upload_catalog.html", {"form":form} )

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
    specimen_list = Specimen.objects.filter(code__exact = specimen_code)
    specimen_result = get_object_or_404(Specimen, pk=specimen_code)
    template = loader.get_template("inventory/bootstrap/specimen.html")

    if specimen_result:
        try:
            spectra = Spectrum.objects.filter(specimen__code = specimen_result.code)
            context = {
                "specimen": specimen_result,
                "spectra": spectra
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

def get_method_by_name(cls, method_name):
    # Check if the method exists in the class
    if hasattr(cls, method_name) and callable(getattr(cls, method_name)):
        # Return the method
        print("success")
        return getattr(cls, method_name)
    else:
        # Method doesn't exist or is not callable
        return None

def spectra(request):
    specimen_list = Specimen.objects.filter(code__exact = specimen_code)
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


def handle_uploaded_spectrum(f):
    def responses(str):
        re1 = "\d+/(\d+,\d+) \d+,\d+/(\d+,\d+)"
        p = re.compile(re1)
        m= p.match(str)
        if m:
            return float(m.group(1).replace(",", ".")),float(m.group(2).replace(",", "."))
        else:
            return 0,0
    def attenuator_settings(str):
        re1 = "S:(\d+,\d+) R:(\d+,\d+)"
        p = re.compile(re1)
        m= p.match(str)
        if m:
            return float(m.group(1).replace(",", ".")),float(m.group(2).replace(",", "."))
        else:
            return 0,0
    def slit_pmt_aperture(str):
        re1 = "\d+/servo \d+,\d+/(\d+,\d+)"
        p = re.compile(re1)
        m= p.match(str)
        if m:
            return Decimal(m.group(1).replace(",", "."))
        else:
            return 0
    #f = open(file_location)
    metadata = {}
    df = pd.DataFrame()
    with open("temp_spectrum.txt", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    with open("temp_spectrum.txt") as data_file:
        for index, row in enumerate(data_file): #0-89

            row_str = row.strip()
            if index +1 == 3: #Filename and extension
                metadata["filename"]= row_str
            if index + 1 == 4: #date DD/MM/YYYY
                metadata["date"]= row_str
            if index + 1 == 5:#Time HH:MM:SS.SS
                metadata["time"]= row_str
            if index + 1 == 8:#user
                metadata["user"]= row_str
            if index + 1 == 9:#description
                metadata["description"]= row_str
            if index + 1 == 10:#minimum wavelength
                metadata["minimum_wavelength"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 12:#equipment name
                metadata["equipment"]= row_str
            if index + 1 == 13:#equipment series
                metadata["series"]= row_str
            if index + 1 == 14:#data visualizer version, equipment version, date and time
                metadata["software"]= row_str
            if index + 1 == 21:#Operating mode
                metadata["operating_mode"]= row_str
            if index + 1 == 22: #Number of cycles
                metadata["cycles"]= row_str
            if index + 1 == 32: #range/servo
                metadata["slit_pmt"]= slit_pmt_aperture(row_str)
            if index + 1 == 33:
                metadata["response_ingaas"], metadata["response_pmt"]= responses(row_str)
            if index + 1 == 35: #pmt gain, if 0 is automatic
                metadata["pmt_gain"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 36: #InGaAs detector gain
                metadata["ingaas_gain"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 42:#monochromator wavelength nm
                metadata["monochromator_change"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 43:#lamp change wavelength
                metadata["lamp_change"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 44:#pmt wavelength
                metadata["pmt_change"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 45:#beam selector
                metadata["beam_selector"]= row_str
            if index + 1 == 46:
                metadata["cbm"]= row_str
            if index + 1 == 47: #cbd status, on/off
                metadata["cbd_status"]= row_str
            if index + 1 == 48: #attenuator percentage
                metadata["attenuator_percentage_sample"], metadata["attenuator_percentage_reference"]= attenuator_settings(row_str)
            if index + 1 == 49:
                metadata["polarizer"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 80:
                metadata["units"]= row_str
            if index + 1 == 81:
                metadata["measuring_mode"]= row_str
            if index + 1 == 84:
                metadata["maximum_wavelength"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 85:
                metadata["step"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 86:
                metadata["number_of_datapoints"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 88:
                metadata["maximum_measurement"]= Decimal(row_str.replace(",", "."))
            if index + 1 == 89:
                metadata["minimum_measurement"]= Decimal(row_str.replace(",", "."))
            if index +1 == 90:
                break

        df = pd.read_csv(data_file, sep="\t", decimal =",", names=["wavelength", metadata["measuring_mode"]])
        return metadata, df

@login_required
def upload_spectra(request):
    template = loader.get_template("inventory/bootstrap/upload_spectra.html")
    context = {}
    if request.method == "POST":
        print("°°°°°°°°°°°Upload spectra Method is post°°°°°°°°°°°°°°°°°°°°")
        form= SpectrumForm(request.POST, request.FILES)


        if form.is_valid():
            print("°°°°°°°°°°°form is valid°°°°°°°°°°°°°°°°°°°°")

            new_spectrum = form.save()
            metadata, df = handle_uploaded_spectrum(request.FILES["file"])
            print(df)
            print("fieldnames")
            #print(dir(Spectrum))
            field_names = [f.name for f in Spectrum._meta.get_fields()]
            for field in field_names:
                try:
                    if metadata[field]:
                        setattr(new_spectrum,field,metadata[field])
                except Exception as e:
                    print("Exception:")
                    print(e)
            print(new_spectrum.operating_mode)

            #Create and save image
            fig = create_svg_image(df, metadata)
            img_filename = new_spectrum.filename.replace(".ASC", "temp.svg")
            fig.savefig(img_filename)


            with open(img_filename, "rb") as image_file:
                new_spectrum.image = UploadedFile(file=image_file)
                new_spectrum.save()

            new_spectrum.save()
            print(new_spectrum.image)
            messages.add_message(request, messages.INFO, f"Spectrum uploaded sucessfully")
    else:
        form = SpectrumForm()
        return render(request,"inventory/bootstrap/upload_spectra.html", {"form":form} )

    return render(request,"inventory/bootstrap/upload_spectra.html", {"form":form} )

@login_required
def download_spectrum(request,spectrum_id):
    spectrum_to_download = Spectrum.objects.get(id = spectrum_id)
    filename = spectrum_to_download.filename
    response = HttpResponse(spectrum_to_download.file, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename=%s' % filename

    return response

def create_svg_image(df, metadata):
    ax = df.plot(x="wavelength", y = metadata["measuring_mode"], grid = True, figsize = (15,10))
    fig = ax.get_figure()

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(metadata["measuring_mode"])
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = 'temp_plot.svg'

    return fig
