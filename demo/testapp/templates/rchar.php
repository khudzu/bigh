{% extends 'cruds/base.html' %}

{% block content %}
{% block title %}Test with Highchart{% endblock %}
<br>

<script type="text/javascript" src="static/code/highcharts.js"></script>
<script type="text/javascript" src="static/code/highcharts-3d.js"></script>
<?php

$connect = mysqli_connect('localhost','root','johndoe','demo');
$sql = "SELECT * FROM employee order by id DESC";
$result = $connect->query($sql);

$arrayFramework = array();
$arrayNilai = array();
if ($result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $arrayFramework[] = '"'.$row['ename'].'"';
        $arrayNilai[] = $row['econtact'];
    }
}
?>

{% endblock content %}
