# Deployment

This app is a static Vite build hosted on `merah`.

## Current Status

- DNS for `padic-logic.symmachus.org` resolves through Cloudflare.
- The static vhost directory has been provisioned on `merah`.
- The current production build has been synced to `htdocs`.
- The proposed `httpd.conf` server block validates with `httpd -n`.
- Public activation still requires adding the server block below to `/etc/httpd.conf` and reloading `httpd`.

## Manual Deploy

```bash
npm run deploy:merah
```

The script runs the production build and syncs `dist/` to:

```text
padiclogic@merah:/var/www/vhosts/padic-logic.symmachus.org/htdocs/
```

## Vhost

The static vhost directory has been provisioned on `merah`:

```text
/var/www/vhosts/padic-logic.symmachus.org/htdocs
```

The `htdocs` directory is owned by `padiclogic:padiclogic`.

Add this server block to `/etc/httpd.conf` on `merah.cassia.ifost.org.au`, then validate and reload `httpd`:

```conf
server "padic-logic.symmachus.org" {
	log style combined
	directory { auto index }
	listen on $listen_addr port 80
	root "/vhosts/padic-logic.symmachus.org/htdocs"
}
```

Validation commands:

```bash
doas httpd -n -f /etc/httpd.conf
doas rcctl reload httpd
```
