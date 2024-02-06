from django import forms

class SpecimenSearchForm(forms.Form):

    code = forms.CharField(label="code", required=False)

    old_code  = forms.CharField(label="old_code", required=False)
    label  = forms.CharField(label="label", required=False)

    genus  = forms.CharField(label="genus", required=False)
    species = forms.CharField(label="species", required=False)
    sex_code = forms.CharField(label="sex_code", required=False)

    notes = forms.CharField(label="notes", required=False)

    country = forms.CharField(label="country", required=False)
    location  = forms.CharField(label="location", required=False)
    location_code  = forms.CharField(label="location_code", required=False)
    province = forms.CharField(label="province", required=False)

    latitude = forms.CharField(label="latitude", required=False)
    longitude = forms.CharField(label="longitude", required=False)
    elevation = forms.CharField(label="elevation", required=False)

    refrigerator= forms.CharField(label="refrigerator", required=False)
    tray = forms.CharField(label="tray", required=False)
    row = forms.CharField(label="row", required=False)
    column  = forms.CharField(label="column", required=False)

    collection_day = forms.CharField(label="collection_day", required=False)
    collection_month = forms.CharField(label="collection_month", required=False)
    collection_year = forms.CharField(label="collection_year", required=False)
    death_date = forms.CharField(label="death_date", required=False)


    light_dark = forms.CharField(label="light_dark", required=False)
    histology_location = forms.CharField(label="histology_location", required=False)
    histology_stage_performed = forms.CharField(label="histology_stage_performed", required=False)
    histology_stage_next_up = forms.CharField(label="histology_stage_next_up", required=False)
    rna_location = forms.CharField(label="rna_location", required=False)
    etoh_voucher = forms.CharField(label="etoh_voucher", required=False)
    elytron = forms.CharField(label="elytron", required=False)
    purpose= forms.CharField(label="purpose", required=False)