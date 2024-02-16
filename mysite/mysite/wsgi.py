"""
WSGI config for mysite project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

#import settings
from django.core.wsgi import get_wsgi_application
#from whitenoise import WhiteNoise
#from mysite import MyWSGIApp

#application = MyWSGIApp()
#application = WhiteNoise(application, root=settings.STATIC_ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")

application = get_wsgi_application()
#application = DjangoWhiteNoise(application)
