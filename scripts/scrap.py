from bs4 import BeautifulSoup as bs
import requests

DOMAIN = 'http://community.tuxguitar.com.ar'
URL = 'http://community.tuxguitar.com.ar/popular/page/'
FILETYPE = '.tg'

nPages = 451

n = 1

for i in range(1, nPages+1):
  soup = bs(requests.get(URL + i.__str__()).text, 'html.parser')
  div = soup.find_all('div', attrs={'class', 'grid_3'})
  a = soup.find_all('a', attrs={'class', 'various iframe lightbox-video'})
  
  
  for link in a:
    link_dl = (DOMAIN + link.get('href')).replace("watch","download")
    print(link_dl)    
    myfile = requests.get(link_dl)
    open('D:/Documents/Ensim/S9b/ISP/isp/scraped_tab/tab_' + str(n) + '.tg', 'wb').write(myfile.content)
    n += 1