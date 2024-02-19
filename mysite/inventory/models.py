from django.db import models
import datetime
from django.utils import timezone


class Specimen(models.Model):
    code = models.CharField(max_length=150, primary_key=True)
    label  = models.CharField(max_length=150, blank=True)
    notes = models.CharField(max_length=500, blank=True)
    old_code  = models.CharField(max_length=150, blank=True)
    collection_day = models.CharField(max_length=150, blank=True)
    collection_month = models.CharField(max_length=150, blank=True)
    collection_year = models.CharField(max_length=150, blank=True)
    death_date = models.CharField(max_length=45, blank=True)
    sex_code = models.CharField(max_length=45, blank=True)
    refrigerator= models.CharField(max_length=150, blank=True)
    tray = models.CharField(max_length=150, blank=True)
    row = models.CharField(max_length=150, blank=True)
    location_code  = models.CharField(max_length=150, blank=True)
    column  = models.CharField(max_length=150, blank=True)
    location  = models.CharField(max_length=150, blank=True)
    genus  = models.CharField(max_length=150, blank=True)
    species = models.CharField(max_length=150, blank=True)
    country = models.CharField(max_length=150, blank=True)
    province = models.CharField(max_length=150, blank=True)
    latitude = models.CharField(max_length=150, blank=True)
    longitude = models.CharField(max_length=150, blank=True)
    elevation = models.CharField(max_length=150, blank=True)
    light_dark = models.CharField(max_length=150, blank=True)
    histology_location = models.CharField(max_length=150, blank=True)
    histology_stage_performed = models.CharField(max_length=150, blank=True)
    histology_stage_next_up = models.CharField(max_length=150, blank=True)
    rna_location = models.CharField(max_length=150, blank=True)
    etoh_voucher = models.CharField(max_length=150, blank=True)
    elytron = models.CharField(max_length=150, blank=True)
    purpose= models.CharField(max_length=150, blank=True)
    def __str__(self):
        return str(self.code)

class Spectrum(models.Model):
    #id = models.AutoField(primary_key = True, null=False, blank=False)
    filename = models.CharField(max_length=150,null=True)
    date  = models.CharField(max_length=150, blank=True,null=True)
    time = models.CharField(max_length=500, blank=True,null=True)
    user  = models.CharField(max_length=150, blank=True,null=True)
    description = models.CharField(max_length=150, blank=True,null=True)
    minimum_wavelength = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    equipment = models.CharField(max_length=150, blank=True,null=True)
    series = models.CharField(max_length=45, blank=True,null=True)
    software = models.CharField(max_length=250, blank=True,null=True)
    operating_mode= models.CharField(max_length=150, blank=True,null=True)
    cycles = models.CharField(max_length=150, blank=True,null=True)
    slit_pmt = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    response_ingaas  = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    response_pmt  = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    pmt_gain  = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    ingaas_gain  = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    monochromator_change = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    lamp_change = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    pmt_change = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    beam_selector = models.CharField(max_length=150, blank=True,null=True)
    cbm = models.CharField(max_length=150, blank=True,null=True)
    cbd_status = models.CharField(max_length=150, blank=True,null=True)
    attenuator_percentage_sample = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    attenuator_percentage_reference = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    polarizer = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    units = models.CharField(max_length=150, blank=True,null=True)
    measuring_mode = models.CharField(max_length=150, blank=True,null=True)
    maximum_wavelength = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    step = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    number_of_datapoints = models.CharField(max_length=150, blank=True,null=True)
    maximum_measurement = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    minimum_measurement = models.DecimalField(max_length=150, blank=True, max_digits = 19, decimal_places = 3,null=True)
    notes= models.CharField(max_length=250, blank=True, help_text = "Please add particular notes for this spectrum: ")
    specimen = models.ForeignKey(Specimen, on_delete = models.SET_NULL, blank=True, null=True)
    file = models.FileField(upload_to="uploads/spectra/%Y/%m/%d/")
    image = models.ImageField(upload_to="images", blank=True, null=True)
    #spectra/%Y/%m/%d/
    def __str__(self):
        def get_specimen_name():
            if self.specimen and self.measuring_mode:
                return str(self.specimen) + " " + str(self.measuring_mode)
            try:
                filename= str(self.id)
                return filename
            except:
                return "No filename"
        return get_specimen_name()

        #return "spectrum"
