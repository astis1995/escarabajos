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
        return "{:04d}".format(int(self.code))
