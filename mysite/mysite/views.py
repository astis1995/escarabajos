from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.template import loader
from django.views import generic
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from . import forms
from django.contrib import messages
import pandas as pd
