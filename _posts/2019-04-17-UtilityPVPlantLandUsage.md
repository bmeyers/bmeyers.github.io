---
layout: post
mathjax: true
title: Land Usage for Utility Scale PV Power Plants
tags: [land usage, pv industry, data scraping, python tricks]
---
 _Scraping data from Wikipedia to investigate how much land a utility PV power plant requires_

 <style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        padding: 10px;
        }
</style>

I received a message from an old friend this afternoon who said, "Random question. How big of site would you guess is required for a 1 megawatt solar field?" To which I responded, in classic Ph.D. fashion, "Well, that's complicated." 

As you might guess, the answer depends heavily on the cell, module, and mounting/tracking technologies used at the power plant. Obviously, a plant built with 25% efficient modules will use less land than a plant built with 15% efficient modules for the same overall capacity. You also need to consider design decisions like [ground cover ratio](https://www.researchgate.net/figure/Ground-coverage-ratio-GCR-is-the-ratio-of-module-area-to-land-area-or-the-ratio-of_fig1_304106060) and many others to exactly estimate this quantity.

The "Suncyclopedia" [states that](http://www.suncyclopedia.com/en/area-required-for-solar-pv-power-plants/) "A simple rule of thumb is to take 100 sqft for every 1kW of solar panels." But to be honest, I did not trust that number! So, I did a little more digging. As it turns out, Wikipedia helpfully provides a [list of photovoltaic power stations that are larger than 200 megawatts in current net capacity](https://en.wikipedia.org/wiki/List_of_photovoltaic_power_stations), which includes nameplate capacity and total land usage for most of the listed power plants.

Having never actually scraped data from a Wikipedia table before, I figured this was a great opportunity to try out a new Python skill, while doing a bit of light research and data analysis. I used `requests` and `Beautiful Soup` to extract the table from Wikipedia and `pandas` to turn the raw html data into a table for analysis.

We begin with the imports we'll need:


{% highlight python%}
import requests
from bs4 import BeautifulSoup
import pandas as pd
{% endhighlight %}

First things first, set up the HTML request, parse the HTML response, and extract the table.


{% highlight python%}
website_url = requests.get('https://en.wikipedia.org/wiki/List_of_photovoltaic_power_stations').text
soup = BeautifulSoup(website_url, 'lxml')
my_table = soup.find('table', {'class': 'wikitable sortable'})
{% endhighlight %}

Tables in Wikipedia tend to have references in the cell text, which is annoying if the cell is supposed to have a float value. Finding and removing the references later can be a hassle, because the references are numeric as is the data we are looking for (and I'm not that proficient at regex). Luckily, `BeautifulSoup` makes searching and modifying HTML trees exceptionally easy. In the cell below, we search all cells in the table and remove all examples of the `reference` class.


{% highlight python%}
for row in my_table.findAll('tr'):
    cells = row.findAll(['th', 'td'])
    for cell in cells:
        references = cell.findAll('sup', {'class': 'reference'})
        if references:
            for ref in references:
                ref.extract()
{% endhighlight %}

`pandas` has all for data I/O needs covered, and comes with an HTML reader. We simply convert the HTML tree to a string and pass it to `pandas` to make a data frame out of.


{% highlight python%}
df = pd.read_html(str(my_table), header=0)[0]
{% endhighlight %}

Now we just need to clean up some column names and data types. Some of the entries in the `Capacity` column contain an asterisk character (`*`) as explained on the Wikipedia page. As with the references, we need to remove these characters to isolate the numerica data. The second to last line below strips all non-numeric characters from the `Capacity` column.


{% highlight python%}
cols = list(df.columns)
cols[3] = 'Capacity'
cols[4] = 'YearlyEnergy'
cols[5] = 'LandSize'
df.columns = cols
df['Capacity'] = df['Capacity'].str.extract('(\d+)', expand=False)
df = df.astype({'LandSize': float, 'Capacity': float})
{% endhighlight %}

And now we have successfully converted the table on Wikipedia to a useable data frame!


{% highlight python%}
df.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Country</th>
      <th>Location</th>
      <th>Capacity</th>
      <th>YearlyEnergy</th>
      <th>LandSize</th>
      <th>Year</th>
      <th>Remarks</th>
      <th>Ref</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tengger Desert Solar Park</td>
      <td>China</td>
      <td>37°33′00″N 105°03′14″E﻿ / ﻿37.55000°N 105.05389°E</td>
      <td>1547.0</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>2016.0</td>
      <td>1,547 MW solar power was installed in Zhongwei...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pavagada Solar Park</td>
      <td>India</td>
      <td>14°05′49″N 77°16′13″E﻿ / ﻿14.09694°N 77.27028°E</td>
      <td>1400.0</td>
      <td>NaN</td>
      <td>53.0</td>
      <td>2019.0</td>
      <td>In Karnataka state, total planned capacity 2,0...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bhadla Solar Park</td>
      <td>India</td>
      <td>27°32′22.81″N 71°54′54.91″E﻿ / ﻿27.5396694°N 7...</td>
      <td>1365.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>2018.0</td>
      <td>The park is proposed to have a capacity of 2,2...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kurnool Ultra Mega Solar Park</td>
      <td>India</td>
      <td>15°40′53″N 78°17′01″E﻿ / ﻿15.681522°N 78.283749°E</td>
      <td>1000.0</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>2017.0</td>
      <td>1000 MW operational as of December 2017</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Datong Solar Power Top Runner Base</td>
      <td>China</td>
      <td>40°04′25″N 113°08′12″E﻿ / ﻿40.07361°N 113.1366...</td>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016.0</td>
      <td>1 GW Phase I completed in June 2016. Total cap...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



And now, let's answer my friend's original question and check if the simple rule of thumb is correct. The data in the table is in terms of MW and square kilometers, so we'll need to change our units to kW and square feet to compare to the given rule of thumb.


{% highlight python%}
land_usage = (df['LandSize'] * 1.076e+7 / df['Capacity'] / 1000).dropna()
plt.figure(figsize=(10,6))
plt.hist(land_usage, bins=20)
plt.xlabel('Square feet per kW')
plt.ylabel('System Count')
title1 = 'Land usage for solar power plants, exracted from:\n'
title2 = 'https://en.wikipedia.org/wiki/List_of_photovoltaic_power_stations'
plt.title(title1 + title2)
plt.axvline(100, ls='--', color='r', label='rule of thumb: 100 ft^2/kW')
med = np.median(land_usage)
plt.axvline(med, ls=':', color='orange', label='median:           {:.0f} ft^2/kW'.format(med))
plt.legend();
{% endhighlight %}

![png]({{ "assets/LandUsage_13_0.png" | absolute_url}})


So, we see that the median value for this set of power plants is more than three times larger than the standard rule of thumb!
