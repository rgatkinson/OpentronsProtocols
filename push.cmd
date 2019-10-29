::
:: push.cmd
::
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org pip uninstall -y rgatkinson-opentrons-enhancements
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org rm -rf /tmp/packaging
scp -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no -r packaging root@clarence.lutzlab.org:/tmp
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org pip install --no-index /tmp/packaging
