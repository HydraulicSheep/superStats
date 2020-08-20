# Linear Regression: Simulation and Calculation
by [HydraulicSheep](https://github.com/HydraulicSheep)

In today's landscape of overwhelming developments in 'Machine Learning' and 'AI', it's easy to forget about the simple powerhouses of the statistics world. One of these is ***Linear Regression***.

References:

http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm **[1]**

https://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html **[2]**

http://seismo.berkeley.edu/~kirchner/eps_120/Toolkits/Toolkit_10.pdf **[3]**

## Simple Linear Regression

Linear regression is simply the fitting of a linear model to a dataset - (making the assumption it is linear). Here, standard linear equations are used.

$$\textbf{y} = mx + c$$

Now, for nearly all datasets (except pure straight lines), it is impossible to find a straight line that passes through every point - that's the whole point of using regression. Therefore, we introduce an **'external error'** term to which we attribute differences between our observation and the quantity's true value.

$$\textbf{y}_i= mx_i + c + ε_i$$

This represents just one datapoint in the data set - $\{(x_1,y_1),(x_2,y_2)...(x_n,y_n)\}$. Here, m and c are the constants we are seeking - the gradient and y-intercept that define the regression line.

So, rearranging, we get: $$\hat{ε}_i = \textbf{y}_i - mx_i - c$$

Note that the switch from $ε_i$ to $\hat{ε}_i$ is intentional: the hat refers to a ***residual*** - the difference between the observation and model value - whereas the non-hatted version refers to an ***error*** - the difference between the observation and true value. The model aims to minimize these ***residuals***

So, now we have established our **model** - A **Simple Linear Regression**

The next thing we need is an approach to determine these parameters - an **estimator**.

## Ordinary Least Squares

**Ordinary Least Squares** (**OLS**) is just one (albeit very popular) technique for determining the parameters of our linear regression model. It states the following:


The best parameters are those which minimise the ***sum of the squared differences*** between each dependent variable datapoint and the y-value estimated by the model.

More formally we want to minimize:

$$\sum_{i=1}^{n} (y_i - c - mx_i)^2 $$

So, an optimization problem. Let's try to simulate a solution before delving into the maths to see if there's an easier approach (there is):


```python
import numpy as np
import pandas as pd
import scipy.optimize
import altair as alt
import random
```

Let's load some data! This dataset [2] includes two points about a number of mammal species - their **Brain Weight** and **Body Weight**.


```python
f = open('mammals.txt')
data = []
for line in f:
    x = list(map(float,line.split()))
    data.append(x)
    
f.close()
data = np.array(data)
dataframe = pd.DataFrame(data=data[:,1:],index=data[:,0],columns=["Brain Weight","Body Weight"])
```


```python
print(dataframe.head(5))
```

         Brain Weight  Body Weight
    1.0         3.385         44.5
    2.0         0.480         15.5
    3.0         1.350          8.1
    4.0       465.000        423.0
    5.0        36.330        119.5


And let's plot the data:


```python
alt.renderers.enable('html')
alt.Chart(dataframe).mark_point().encode(
    x=alt.X(field='Brain Weight',type='quantitative'),
    y=alt.Y(field='Body Weight', type='quantitative', sort='x')
)
```





<div id="altair-viz-3ae8ea5ca7134a509aa0d757f1312024"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-3ae8ea5ca7134a509aa0d757f1312024") {
      outputDiv = document.getElementById("altair-viz-3ae8ea5ca7134a509aa0d757f1312024");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-038fe16cf7fa3ac06dc0e6e21d97a694"}, "mark": "point", "encoding": {"x": {"type": "quantitative", "field": "Brain Weight"}, "y": {"type": "quantitative", "field": "Body Weight", "sort": "x"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-038fe16cf7fa3ac06dc0e6e21d97a694": [{"Brain Weight": 3.385, "Body Weight": 44.5}, {"Brain Weight": 0.48, "Body Weight": 15.5}, {"Brain Weight": 1.35, "Body Weight": 8.1}, {"Brain Weight": 465.0, "Body Weight": 423.0}, {"Brain Weight": 36.33, "Body Weight": 119.5}, {"Brain Weight": 27.66, "Body Weight": 115.0}, {"Brain Weight": 14.83, "Body Weight": 98.2}, {"Brain Weight": 1.04, "Body Weight": 5.5}, {"Brain Weight": 4.19, "Body Weight": 58.0}, {"Brain Weight": 0.425, "Body Weight": 6.4}, {"Brain Weight": 0.101, "Body Weight": 4.0}, {"Brain Weight": 0.92, "Body Weight": 5.7}, {"Brain Weight": 1.0, "Body Weight": 6.6}, {"Brain Weight": 0.005, "Body Weight": 0.14}, {"Brain Weight": 0.06, "Body Weight": 1.0}, {"Brain Weight": 3.5, "Body Weight": 10.8}, {"Brain Weight": 2.0, "Body Weight": 12.3}, {"Brain Weight": 1.7, "Body Weight": 6.3}, {"Brain Weight": 2547.0, "Body Weight": 4603.0}, {"Brain Weight": 0.023, "Body Weight": 0.3}, {"Brain Weight": 187.1, "Body Weight": 419.0}, {"Brain Weight": 521.0, "Body Weight": 655.0}, {"Brain Weight": 0.785, "Body Weight": 3.5}, {"Brain Weight": 10.0, "Body Weight": 115.0}, {"Brain Weight": 3.3, "Body Weight": 25.6}, {"Brain Weight": 0.2, "Body Weight": 5.0}, {"Brain Weight": 1.41, "Body Weight": 17.5}, {"Brain Weight": 529.0, "Body Weight": 680.0}, {"Brain Weight": 207.0, "Body Weight": 406.0}, {"Brain Weight": 85.0, "Body Weight": 325.0}, {"Brain Weight": 0.75, "Body Weight": 12.3}, {"Brain Weight": 62.0, "Body Weight": 1320.0}, {"Brain Weight": 6654.0, "Body Weight": 5712.0}, {"Brain Weight": 3.5, "Body Weight": 3.9}, {"Brain Weight": 6.8, "Body Weight": 179.0}, {"Brain Weight": 35.0, "Body Weight": 56.0}, {"Brain Weight": 4.05, "Body Weight": 17.0}, {"Brain Weight": 0.12, "Body Weight": 1.0}, {"Brain Weight": 0.023, "Body Weight": 0.4}, {"Brain Weight": 0.01, "Body Weight": 0.25}, {"Brain Weight": 1.4, "Body Weight": 12.5}, {"Brain Weight": 250.0, "Body Weight": 490.0}, {"Brain Weight": 2.5, "Body Weight": 12.1}, {"Brain Weight": 55.5, "Body Weight": 175.0}, {"Brain Weight": 100.0, "Body Weight": 157.0}, {"Brain Weight": 52.16, "Body Weight": 440.0}, {"Brain Weight": 10.55, "Body Weight": 179.5}, {"Brain Weight": 0.55, "Body Weight": 2.4}, {"Brain Weight": 60.0, "Body Weight": 81.0}, {"Brain Weight": 3.6, "Body Weight": 21.0}, {"Brain Weight": 4.288, "Body Weight": 39.2}, {"Brain Weight": 0.28, "Body Weight": 1.9}, {"Brain Weight": 0.075, "Body Weight": 1.2}, {"Brain Weight": 0.122, "Body Weight": 3.0}, {"Brain Weight": 0.048, "Body Weight": 0.33}, {"Brain Weight": 192.0, "Body Weight": 180.0}, {"Brain Weight": 3.0, "Body Weight": 25.0}, {"Brain Weight": 160.0, "Body Weight": 169.0}, {"Brain Weight": 0.9, "Body Weight": 2.6}, {"Brain Weight": 1.62, "Body Weight": 11.4}, {"Brain Weight": 0.104, "Body Weight": 2.5}, {"Brain Weight": 4.235, "Body Weight": 50.4}]}}, {"mode": "vega-lite"});
</script>



Hmmm... It looks like the massive values are making it difficult to see the smaller ones. Let's only display a few values, to make viewing them easier.


```python
alt.Chart(dataframe).mark_point(clip=True).encode(
    x=alt.X(field='Brain Weight',type='quantitative',scale=alt.Scale(domain=(0,4))),
    y=alt.Y(field='Body Weight', type='quantitative', sort='x',scale=alt.Scale(domain=(0, 30)))
)
```





<div id="altair-viz-29f6d50ddba640c6bb871605c8229d02"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-29f6d50ddba640c6bb871605c8229d02") {
      outputDiv = document.getElementById("altair-viz-29f6d50ddba640c6bb871605c8229d02");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-038fe16cf7fa3ac06dc0e6e21d97a694"}, "mark": {"type": "point", "clip": true}, "encoding": {"x": {"type": "quantitative", "field": "Brain Weight", "scale": {"domain": [0, 4]}}, "y": {"type": "quantitative", "field": "Body Weight", "scale": {"domain": [0, 30]}, "sort": "x"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-038fe16cf7fa3ac06dc0e6e21d97a694": [{"Brain Weight": 3.385, "Body Weight": 44.5}, {"Brain Weight": 0.48, "Body Weight": 15.5}, {"Brain Weight": 1.35, "Body Weight": 8.1}, {"Brain Weight": 465.0, "Body Weight": 423.0}, {"Brain Weight": 36.33, "Body Weight": 119.5}, {"Brain Weight": 27.66, "Body Weight": 115.0}, {"Brain Weight": 14.83, "Body Weight": 98.2}, {"Brain Weight": 1.04, "Body Weight": 5.5}, {"Brain Weight": 4.19, "Body Weight": 58.0}, {"Brain Weight": 0.425, "Body Weight": 6.4}, {"Brain Weight": 0.101, "Body Weight": 4.0}, {"Brain Weight": 0.92, "Body Weight": 5.7}, {"Brain Weight": 1.0, "Body Weight": 6.6}, {"Brain Weight": 0.005, "Body Weight": 0.14}, {"Brain Weight": 0.06, "Body Weight": 1.0}, {"Brain Weight": 3.5, "Body Weight": 10.8}, {"Brain Weight": 2.0, "Body Weight": 12.3}, {"Brain Weight": 1.7, "Body Weight": 6.3}, {"Brain Weight": 2547.0, "Body Weight": 4603.0}, {"Brain Weight": 0.023, "Body Weight": 0.3}, {"Brain Weight": 187.1, "Body Weight": 419.0}, {"Brain Weight": 521.0, "Body Weight": 655.0}, {"Brain Weight": 0.785, "Body Weight": 3.5}, {"Brain Weight": 10.0, "Body Weight": 115.0}, {"Brain Weight": 3.3, "Body Weight": 25.6}, {"Brain Weight": 0.2, "Body Weight": 5.0}, {"Brain Weight": 1.41, "Body Weight": 17.5}, {"Brain Weight": 529.0, "Body Weight": 680.0}, {"Brain Weight": 207.0, "Body Weight": 406.0}, {"Brain Weight": 85.0, "Body Weight": 325.0}, {"Brain Weight": 0.75, "Body Weight": 12.3}, {"Brain Weight": 62.0, "Body Weight": 1320.0}, {"Brain Weight": 6654.0, "Body Weight": 5712.0}, {"Brain Weight": 3.5, "Body Weight": 3.9}, {"Brain Weight": 6.8, "Body Weight": 179.0}, {"Brain Weight": 35.0, "Body Weight": 56.0}, {"Brain Weight": 4.05, "Body Weight": 17.0}, {"Brain Weight": 0.12, "Body Weight": 1.0}, {"Brain Weight": 0.023, "Body Weight": 0.4}, {"Brain Weight": 0.01, "Body Weight": 0.25}, {"Brain Weight": 1.4, "Body Weight": 12.5}, {"Brain Weight": 250.0, "Body Weight": 490.0}, {"Brain Weight": 2.5, "Body Weight": 12.1}, {"Brain Weight": 55.5, "Body Weight": 175.0}, {"Brain Weight": 100.0, "Body Weight": 157.0}, {"Brain Weight": 52.16, "Body Weight": 440.0}, {"Brain Weight": 10.55, "Body Weight": 179.5}, {"Brain Weight": 0.55, "Body Weight": 2.4}, {"Brain Weight": 60.0, "Body Weight": 81.0}, {"Brain Weight": 3.6, "Body Weight": 21.0}, {"Brain Weight": 4.288, "Body Weight": 39.2}, {"Brain Weight": 0.28, "Body Weight": 1.9}, {"Brain Weight": 0.075, "Body Weight": 1.2}, {"Brain Weight": 0.122, "Body Weight": 3.0}, {"Brain Weight": 0.048, "Body Weight": 0.33}, {"Brain Weight": 192.0, "Body Weight": 180.0}, {"Brain Weight": 3.0, "Body Weight": 25.0}, {"Brain Weight": 160.0, "Body Weight": 169.0}, {"Brain Weight": 0.9, "Body Weight": 2.6}, {"Brain Weight": 1.62, "Body Weight": 11.4}, {"Brain Weight": 0.104, "Body Weight": 2.5}, {"Brain Weight": 4.235, "Body Weight": 50.4}]}}, {"mode": "vega-lite"});
</script>



Well, there we go. You may notice a vague linear trend but we want to test for this mathematically.

We can also plot the graph with a ***log-log scale*** to view all the points on one graph.


```python
c1 = alt.Chart(dataframe).mark_point().encode(
    x=alt.X(field='Brain Weight',type='quantitative',scale=alt.Scale(type='log')),
    y=alt.Y(field='Body Weight', type='quantitative', sort='x',scale=alt.Scale(type='log'))
)
c1
```





<div id="altair-viz-0da3a7e4e939440d9d674bff872fe8ee"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0da3a7e4e939440d9d674bff872fe8ee") {
      outputDiv = document.getElementById("altair-viz-0da3a7e4e939440d9d674bff872fe8ee");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-038fe16cf7fa3ac06dc0e6e21d97a694"}, "mark": "point", "encoding": {"x": {"type": "quantitative", "field": "Brain Weight", "scale": {"type": "log"}}, "y": {"type": "quantitative", "field": "Body Weight", "scale": {"type": "log"}, "sort": "x"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-038fe16cf7fa3ac06dc0e6e21d97a694": [{"Brain Weight": 3.385, "Body Weight": 44.5}, {"Brain Weight": 0.48, "Body Weight": 15.5}, {"Brain Weight": 1.35, "Body Weight": 8.1}, {"Brain Weight": 465.0, "Body Weight": 423.0}, {"Brain Weight": 36.33, "Body Weight": 119.5}, {"Brain Weight": 27.66, "Body Weight": 115.0}, {"Brain Weight": 14.83, "Body Weight": 98.2}, {"Brain Weight": 1.04, "Body Weight": 5.5}, {"Brain Weight": 4.19, "Body Weight": 58.0}, {"Brain Weight": 0.425, "Body Weight": 6.4}, {"Brain Weight": 0.101, "Body Weight": 4.0}, {"Brain Weight": 0.92, "Body Weight": 5.7}, {"Brain Weight": 1.0, "Body Weight": 6.6}, {"Brain Weight": 0.005, "Body Weight": 0.14}, {"Brain Weight": 0.06, "Body Weight": 1.0}, {"Brain Weight": 3.5, "Body Weight": 10.8}, {"Brain Weight": 2.0, "Body Weight": 12.3}, {"Brain Weight": 1.7, "Body Weight": 6.3}, {"Brain Weight": 2547.0, "Body Weight": 4603.0}, {"Brain Weight": 0.023, "Body Weight": 0.3}, {"Brain Weight": 187.1, "Body Weight": 419.0}, {"Brain Weight": 521.0, "Body Weight": 655.0}, {"Brain Weight": 0.785, "Body Weight": 3.5}, {"Brain Weight": 10.0, "Body Weight": 115.0}, {"Brain Weight": 3.3, "Body Weight": 25.6}, {"Brain Weight": 0.2, "Body Weight": 5.0}, {"Brain Weight": 1.41, "Body Weight": 17.5}, {"Brain Weight": 529.0, "Body Weight": 680.0}, {"Brain Weight": 207.0, "Body Weight": 406.0}, {"Brain Weight": 85.0, "Body Weight": 325.0}, {"Brain Weight": 0.75, "Body Weight": 12.3}, {"Brain Weight": 62.0, "Body Weight": 1320.0}, {"Brain Weight": 6654.0, "Body Weight": 5712.0}, {"Brain Weight": 3.5, "Body Weight": 3.9}, {"Brain Weight": 6.8, "Body Weight": 179.0}, {"Brain Weight": 35.0, "Body Weight": 56.0}, {"Brain Weight": 4.05, "Body Weight": 17.0}, {"Brain Weight": 0.12, "Body Weight": 1.0}, {"Brain Weight": 0.023, "Body Weight": 0.4}, {"Brain Weight": 0.01, "Body Weight": 0.25}, {"Brain Weight": 1.4, "Body Weight": 12.5}, {"Brain Weight": 250.0, "Body Weight": 490.0}, {"Brain Weight": 2.5, "Body Weight": 12.1}, {"Brain Weight": 55.5, "Body Weight": 175.0}, {"Brain Weight": 100.0, "Body Weight": 157.0}, {"Brain Weight": 52.16, "Body Weight": 440.0}, {"Brain Weight": 10.55, "Body Weight": 179.5}, {"Brain Weight": 0.55, "Body Weight": 2.4}, {"Brain Weight": 60.0, "Body Weight": 81.0}, {"Brain Weight": 3.6, "Body Weight": 21.0}, {"Brain Weight": 4.288, "Body Weight": 39.2}, {"Brain Weight": 0.28, "Body Weight": 1.9}, {"Brain Weight": 0.075, "Body Weight": 1.2}, {"Brain Weight": 0.122, "Body Weight": 3.0}, {"Brain Weight": 0.048, "Body Weight": 0.33}, {"Brain Weight": 192.0, "Body Weight": 180.0}, {"Brain Weight": 3.0, "Body Weight": 25.0}, {"Brain Weight": 160.0, "Body Weight": 169.0}, {"Brain Weight": 0.9, "Body Weight": 2.6}, {"Brain Weight": 1.62, "Body Weight": 11.4}, {"Brain Weight": 0.104, "Body Weight": 2.5}, {"Brain Weight": 4.235, "Body Weight": 50.4}]}}, {"mode": "vega-lite"});
</script>



Wow, that looks surprisingly linear. *BUT...* **Don't be deceived**. Linear regression runs on the underlying data values (rather than their log representations). And relationships of any form other than $ax^{n}$ will differ from their log-log plots E.g. the straight line y=x+10 below:


```python
source = alt.sequence(start=1, stop=10000, step=0.5, as_='x')

alt.Chart(source).mark_line().transform_calculate(
    fx='datum.x+10',
).transform_fold(
    ['fx']
).encode(
    x=alt.X(field='x',type='quantitative',scale=alt.Scale(type='log')),
    y=alt.X(field='value',type='quantitative',scale=alt.Scale(type='log')),
    color='key:N'
)
```





<div id="altair-viz-693d964f6c6144da8c741f2c9c751299"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-693d964f6c6144da8c741f2c9c751299") {
      outputDiv = document.getElementById("altair-viz-693d964f6c6144da8c741f2c9c751299");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"sequence": {"start": 1, "stop": 10000, "step": 0.5, "as": "x"}}, "mark": "line", "encoding": {"color": {"type": "nominal", "field": "key"}, "x": {"type": "quantitative", "field": "x", "scale": {"type": "log"}}, "y": {"type": "quantitative", "field": "value", "scale": {"type": "log"}}}, "transform": [{"calculate": "datum.x+10", "as": "fx"}, {"fold": ["fx"]}], "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json"}, {"mode": "vega-lite"});
</script>



### Approach via Simulated Annealing

Simulated annealing is an optimization technique (like gradient descent or hill-climbing) that helps to search for the global maxima/minima of a function. Compared with gradient descent or hill-climbing specifically, it is able to ***ignore local maxmima/minima*** and just ***find the global optima*** (given slow enough 'cooling'). 

So, let's define a loss function - the function that we want to minimise - to perform Ordinary Least Squares


```python
#We'll define a loss function for ordinary least squares that takes arguments m and c

def loss_function(m,c):
    global dataframe

    y = lambda x: m*x + c
    
    ols = 0
    
    for index, row in dataframe.iterrows():
        ols += (row['Body Weight']-y(row['Brain Weight']))**2
    return ols

```


```python
#Let's try one set of parameters out:
print(loss_function(20000,400))
```

    2.070522112827237e+16


And now we can write a framework for ***Simulated Annealing***:


```python
#Now, there is no way of sampling over all the real numbers, so let's define some bounds for our annealing
#And from simply looking at the data, the gradients nor intercepts cannot be ridiculously high - 
#Let's restrict them to +-50 and +-1000 respectively - simply from a graphical view of the data
bounds = {'m':[-50,50],'c':[-1000,1000]}
```


```python
#Now, set an initial set of parameters for the system:
params = {'m':random.uniform(bounds['m'][0], bounds['m'][1]),'c':random.uniform(bounds['c'][0], bounds['c'][1])}
print(params)
cost = loss_function(params['m'],params['c'])
```

    {'m': 19.669581838767286, 'c': 822.6342542476934}



```python
def getRandomNeighbour(x,bounds,temperature):
    
    #Gets a random neighbour. Max step of 1/8*1/2 = 1*range/16.  
    
    valRange = temperature * (bounds[1] - bounds[0])/8
    d =  valRange * (random.random()-0.5)
    neighbour = x+d
    
    #Restricts neighbour values to within our range.
    if neighbour > bounds[1]:
        neighbour = bounds[1]
    if neighbour < bounds[0]:
        neighbour = bounds[0]
        
    return neighbour

def acceptanceFunction(cost, newCost, temperature):
    if newCost < cost:
        #Always descend if the cost is lower
        return 1
    else:
        #Simulated Annealing Probability must tend to zero as temperature tends to zero.
        #Lower probabilities for greater new costs.
        p = np.exp((cost-newCost)/temperature)
        return p

```

Great! Now that we've got our helper functions, we can execute ***Simulated Annealing*** over a number of iterations:


```python
ITERATIONS = 10000
changes = []
for i in range(ITERATIONS):
    completion = i/ITERATIONS
    temp = max(0.01, min(1, 1 - completion))
    
    newM = getRandomNeighbour(params['m'],bounds['m'],temp)
    newC = getRandomNeighbour(params['c'],bounds['c'],temp)
    
    newCost = loss_function(newM,newC)
    
    
    
    if acceptanceFunction(cost,newCost,temp) > random.random():
        changes.append([i,newCost])
        params['m'] = newM
        params['c'] = newC
        cost = newCost
    
    
```


```python
#Output discarded for being too large.
print(params)
print(loss_function(0.9663325792000524,90.4196277321998))
```

    {'m': 0.9664734151151319, 'c': 91.31034039313383}
    6722264.006349455



```python
ser = pd.Series(changes)
frame = pd.DataFrame(ser.values.tolist(), index=ser.index,columns=["Iterations","Cost"])
print(frame.head(3))
alt.Chart(frame).mark_point().encode(
    x=alt.X(field='Iterations',type='quantitative'),
    y=alt.Y(field='Cost',type='quantitative',scale=alt.Scale(type='log'))
)

```

       Iterations          Cost
    0           1  1.221542e+10
    1           7  8.822840e+09
    2           8  4.642547e+09






<div id="altair-viz-0c18624f884c4cf6a164dd910ab3903c"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0c18624f884c4cf6a164dd910ab3903c") {
      outputDiv = document.getElementById("altair-viz-0c18624f884c4cf6a164dd910ab3903c");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-6c49e906b6ca4fd272738ff082d7e501"}, "mark": "point", "encoding": {"x": {"type": "quantitative", "field": "Iterations"}, "y": {"type": "quantitative", "field": "Cost", "scale": {"type": "log"}}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-6c49e906b6ca4fd272738ff082d7e501": [{"Iterations": 1, "Cost": 12215416320.710657}, {"Iterations": 7, "Cost": 8822839920.72128}, {"Iterations": 8, "Cost": 4642546824.596871}, {"Iterations": 9, "Cost": 1451258120.3004317}, {"Iterations": 10, "Cost": 99780504.54059175}, {"Iterations": 15, "Cost": 32627927.38495094}, {"Iterations": 31, "Cost": 28361591.190102544}, {"Iterations": 86, "Cost": 20454251.535725743}, {"Iterations": 89, "Cost": 10939505.79530641}, {"Iterations": 277, "Cost": 9576329.130038567}, {"Iterations": 344, "Cost": 7829320.874370154}, {"Iterations": 512, "Cost": 7217129.169795612}, {"Iterations": 643, "Cost": 6879137.740920027}, {"Iterations": 718, "Cost": 6835735.409195979}, {"Iterations": 782, "Cost": 6745759.562308909}, {"Iterations": 2233, "Cost": 6738727.902571717}, {"Iterations": 2245, "Cost": 6733058.867785064}, {"Iterations": 7248, "Cost": 6726704.35494863}, {"Iterations": 8429, "Cost": 6723898.650653476}, {"Iterations": 9101, "Cost": 6723810.188895634}, {"Iterations": 9500, "Cost": 6723364.250746809}, {"Iterations": 9558, "Cost": 6722631.981075335}, {"Iterations": 9733, "Cost": 6722304.159040615}, {"Iterations": 9927, "Cost": 6722244.71299463}]}}, {"mode": "vega-lite"});
</script>



Wow! Look at that! After just 10000 iterations (taking a minute or two), the simulation has seemed to hone in on some values pretty clearly. 

$$ y = 0.96647x + 91.310 $$

Let's examine them on the graph:


```python
source = alt.sequence(start=1, stop=10000, step=0.5, as_='x')

c1 = alt.Chart(dataframe).mark_point(clip=True).encode(
    x=alt.X(field='Brain Weight',type='quantitative',scale=alt.Scale(domain=(0, 500))),
    y=alt.Y(field='Body Weight', type='quantitative',scale=alt.Scale(domain=(0, 600)))
)


c2 = alt.Chart(source).mark_line(clip=True).transform_calculate(
    fx='0.9664734151151319*datum.x+91.31034039313383',
).transform_fold(
    ['fx']
).encode(
    x=alt.X(field='x',type='quantitative',),
    y=alt.X(field='value',type='quantitative',),
    color='key:N'
)
alt.layer(c1,c2)
```





<div id="altair-viz-1dd156d9de754e3ebcd78db3dfd99368"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-1dd156d9de754e3ebcd78db3dfd99368") {
      outputDiv = document.getElementById("altair-viz-1dd156d9de754e3ebcd78db3dfd99368");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"data": {"name": "data-038fe16cf7fa3ac06dc0e6e21d97a694"}, "mark": {"type": "point", "clip": true}, "encoding": {"x": {"type": "quantitative", "field": "Brain Weight", "scale": {"domain": [0, 500]}}, "y": {"type": "quantitative", "field": "Body Weight", "scale": {"domain": [0, 600]}}}}, {"data": {"sequence": {"start": 1, "stop": 10000, "step": 0.5, "as": "x"}}, "mark": {"type": "line", "clip": true}, "encoding": {"color": {"type": "nominal", "field": "key"}, "x": {"type": "quantitative", "field": "x"}, "y": {"type": "quantitative", "field": "value"}}, "transform": [{"calculate": "0.9664734151151319*datum.x+91.31034039313383", "as": "fx"}, {"fold": ["fx"]}]}], "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-038fe16cf7fa3ac06dc0e6e21d97a694": [{"Brain Weight": 3.385, "Body Weight": 44.5}, {"Brain Weight": 0.48, "Body Weight": 15.5}, {"Brain Weight": 1.35, "Body Weight": 8.1}, {"Brain Weight": 465.0, "Body Weight": 423.0}, {"Brain Weight": 36.33, "Body Weight": 119.5}, {"Brain Weight": 27.66, "Body Weight": 115.0}, {"Brain Weight": 14.83, "Body Weight": 98.2}, {"Brain Weight": 1.04, "Body Weight": 5.5}, {"Brain Weight": 4.19, "Body Weight": 58.0}, {"Brain Weight": 0.425, "Body Weight": 6.4}, {"Brain Weight": 0.101, "Body Weight": 4.0}, {"Brain Weight": 0.92, "Body Weight": 5.7}, {"Brain Weight": 1.0, "Body Weight": 6.6}, {"Brain Weight": 0.005, "Body Weight": 0.14}, {"Brain Weight": 0.06, "Body Weight": 1.0}, {"Brain Weight": 3.5, "Body Weight": 10.8}, {"Brain Weight": 2.0, "Body Weight": 12.3}, {"Brain Weight": 1.7, "Body Weight": 6.3}, {"Brain Weight": 2547.0, "Body Weight": 4603.0}, {"Brain Weight": 0.023, "Body Weight": 0.3}, {"Brain Weight": 187.1, "Body Weight": 419.0}, {"Brain Weight": 521.0, "Body Weight": 655.0}, {"Brain Weight": 0.785, "Body Weight": 3.5}, {"Brain Weight": 10.0, "Body Weight": 115.0}, {"Brain Weight": 3.3, "Body Weight": 25.6}, {"Brain Weight": 0.2, "Body Weight": 5.0}, {"Brain Weight": 1.41, "Body Weight": 17.5}, {"Brain Weight": 529.0, "Body Weight": 680.0}, {"Brain Weight": 207.0, "Body Weight": 406.0}, {"Brain Weight": 85.0, "Body Weight": 325.0}, {"Brain Weight": 0.75, "Body Weight": 12.3}, {"Brain Weight": 62.0, "Body Weight": 1320.0}, {"Brain Weight": 6654.0, "Body Weight": 5712.0}, {"Brain Weight": 3.5, "Body Weight": 3.9}, {"Brain Weight": 6.8, "Body Weight": 179.0}, {"Brain Weight": 35.0, "Body Weight": 56.0}, {"Brain Weight": 4.05, "Body Weight": 17.0}, {"Brain Weight": 0.12, "Body Weight": 1.0}, {"Brain Weight": 0.023, "Body Weight": 0.4}, {"Brain Weight": 0.01, "Body Weight": 0.25}, {"Brain Weight": 1.4, "Body Weight": 12.5}, {"Brain Weight": 250.0, "Body Weight": 490.0}, {"Brain Weight": 2.5, "Body Weight": 12.1}, {"Brain Weight": 55.5, "Body Weight": 175.0}, {"Brain Weight": 100.0, "Body Weight": 157.0}, {"Brain Weight": 52.16, "Body Weight": 440.0}, {"Brain Weight": 10.55, "Body Weight": 179.5}, {"Brain Weight": 0.55, "Body Weight": 2.4}, {"Brain Weight": 60.0, "Body Weight": 81.0}, {"Brain Weight": 3.6, "Body Weight": 21.0}, {"Brain Weight": 4.288, "Body Weight": 39.2}, {"Brain Weight": 0.28, "Body Weight": 1.9}, {"Brain Weight": 0.075, "Body Weight": 1.2}, {"Brain Weight": 0.122, "Body Weight": 3.0}, {"Brain Weight": 0.048, "Body Weight": 0.33}, {"Brain Weight": 192.0, "Body Weight": 180.0}, {"Brain Weight": 3.0, "Body Weight": 25.0}, {"Brain Weight": 160.0, "Body Weight": 169.0}, {"Brain Weight": 0.9, "Body Weight": 2.6}, {"Brain Weight": 1.62, "Body Weight": 11.4}, {"Brain Weight": 0.104, "Body Weight": 2.5}, {"Brain Weight": 4.235, "Body Weight": 50.4}]}}, {"mode": "vega-lite"});
</script>



We can also try running with a few more iterations:


```python
#Now, set a new initial set of parameters for the system:
params = {'m':random.uniform(bounds['m'][0], bounds['m'][1]),'c':random.uniform(bounds['c'][0], bounds['c'][1])}
cost = loss_function(params['m'],params['c'])

ITERATIONS = 100000
changes = []
for i in range(ITERATIONS):
    completion = i/ITERATIONS
    temp = max(0.01, min(1, 1 - completion))
    
    newM = getRandomNeighbour(params['m'],bounds['m'],temp)
    newC = getRandomNeighbour(params['c'],bounds['c'],temp)
    
    newCost = loss_function(newM,newC)
    
    
    if acceptanceFunction(cost,newCost,temp) > random.random():
        changes.append([i,newCost])
        params['m'] = newM
        params['c'] = newC
        cost = newCost
    
```


```python
print(params)
print(loss_function(0.9659362423807267,91.29398751426505))
```

    {'m': 0.9659362423807267, 'c': 91.29398751426505}
    6722256.498220906


Interesting... This only barely improved our result (cost) but took 10 times longer. There are a number of factors at play:
* Simulated Annealing is a complex process with many different parameters that can be tweaked (e.g. the form of our probability function and range scaling factors) and general optimizations. 
* It is a stochastic (rather than deterministic) process so it's entirely possible to have a bad run/bad initial values and get poor results.

Also of note: We permitted the variables to change within fairly large domains. This means that more iterations are needed to get the same level of granularity for each variable. This is almost like a 'resolution'.

So, the best regression we obtained (the one with the lowest cost) was:

$$ y = 0.96594x + 91.2940 $$


### Approach via Statistical Methods

The statistical method approach involves slightly more maths but in exchange, we receive a far cleaner solution.

A full derivation is provided in **[3]**, but, in summary:


$$ Q = \sum_{i=1}^{n} (y_i - a - bx_i)^2 $$

Q will be minimized when 
$$ \frac{\partial Q}{\partial a} = 0 \quad [1] \quad and \quad \frac{\partial Q}{\partial b} = 0 \quad [2] $$

Expanding and simplifying [1] produces

$$ \hat{a} = \bar{Y} - \hat{b}\bar{X},\space where \space \hat{a},\hat{b}\space are\space minimising \space values$$

Expanding and simplifying [2] produces

$$ \hat{b} = \frac{Covariance(X,Y)}{Variance(X)} $$

Now, covariance is given by the following formula:

$$ Cov(X,Y) = \frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{n} $$ 

And for variance:

$$ Var(X) = \frac{\sum_{i=1}^{n}(x_{i}-\bar{x})^2}{n} $$ 

*Note: These refer to **population variance and covariance**. We are using these because we are performing regression over the whole dataset - a population - (not the observed system).*

So, let's perform the calculation:


```python
xtotal = 0
ytotal = 0
count = 0
covariance = 0
variance = 0
for index, row in dataframe.iterrows():
    xtotal += row['Brain Weight']
    ytotal += row['Body Weight']
    count +=1
xmean = xtotal / count
ymean = ytotal / count
for index, row in dataframe.iterrows():
    x = row['Brain Weight']
    y = row['Body Weight']
    covariance += (x-xmean)*(y-ymean)
    variance += (x-xmean)**2
variance /= count
covariance /= count
b = covariance/variance
print("Variance: " + str(variance))
print("Covariance: " + str(covariance))
print("b (Gradient): " + str(b))
print("a (Intercept): " + str(ymean-b*xmean))
```

    Variance: 795445.0451699513
    Covariance: 768794.7468399063
    b (Gradient): 0.9664963676725763
    a (Intercept): 91.00439620740684



```python
#Loss Value
loss_function(0.9664963676725763,91.00439620740684)
```




    6722239.055504982



And look at that! The proven statistical values match the simulation-derived ones (to 1 decimal place):

$$ y = 0.96594x + 91.2940 \quad Cost: \space 6722256$$

$$ vs. $$

$$ y = 0.96650x + 91.004 \quad Cost: \space 6722239$$

So the cost values only differ by ≈17 which doesn't seem bad given the massive cost values passed over during simulation.

And yes, there are a whole bunch of things I haven't covered here - R^2 Coefficients, Multivariate Linear Regression, Non-Linear Regression - but this notebook is already beginning to drag! 

But even simple, ordinary Linear Regression is a powerful tool; ***Use it Wisely!***

______________
Stay tuned to my ***Github*** - [HydraulicSheep](https://github.com/HydraulicSheep) - for explorations of more great ***Game Theory*** and ***Statistics*** content.
